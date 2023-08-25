import torch
import torch.nn as nn
import torch.nn.functional as func
from math import sqrt, prod

from eventful_transformer.base import ExtendedModule, numeric_tuple
from eventful_transformer.counting import CountedAdd, CountedLinear, CountedMatmul
from eventful_transformer.modules import (
    SimpleSTGTGate,
    TokenBuffer,
    TokenDeltaGate,
    TokenGate,
    MatmulDeltaAccumulator,
    MatmulBuffer,
)
from eventful_transformer.utils import (
    DropPath,
    RelativePositionEmbedding,
    expand_row_index,
)
from utils.image import pad_to_size

LN_EPS = 1e-6


class Block(ExtendedModule):
    """
    Defines a base (non-eventful) Transformer block. Includes a couple
    of extra features: a simple implementation of Adaptive Token
    Sampling (ATS - Fayyaz et al. 2022) and self-attention pooling.
    These features are controlled via the ats_fraction and pool_size
    parameters.
    """

    def __init__(
        self,
        dim,
        heads,
        input_size,
        mlp_ratio,
        ats_fraction=None,
        drop_path_rate=0.0,
        relative_embedding_size=None,
        matmul_2_cast=None,
        pool_size=None,
        window_size=None,
    ):
        """
        :param dim: The number of dimensions in a token
        :param heads: The number of attention heads (None for no
        multi-headed attention)
        :param input_size: The expected size of the inputs in tokens
        :param mlp_ratio: The ratio of the MLP dimensionality to the
        token dimensionality
        :param ats_fraction: The fraction of tokens to retain if
        using Adaptive Token Sampling (ATS)
        :param drop_path_rate: Drop path ratio (for use when training)
        :param relative_embedding_size: The size (in tokens) assumed for
        relative position embeddings
        :param matmul_2_cast: Typecast for the attention-value product
        (None, "float16", or "bfloat16"). Helps save some memory when
        using an A-gate, without a noticeable impact on accuracy.
        :param pool_size: Pooling ratio to use with self-attention
        pooling.
        :param window_size: Self-attention window size (None to use
        global, non-windowed attention).
        """
        super().__init__()
        self.heads = heads
        self.input_size = tuple(input_size)
        if ats_fraction is not None:
            assert pool_size is None
            assert window_size is None
            assert not (ats_fraction < 0.0 or ats_fraction > 1.0)
        assert not (drop_path_rate < 0.0 or drop_path_rate > 1.0)
        assert matmul_2_cast in [None, "float16", "bfloat16"]
        self.ats_fraction = ats_fraction
        self.last_ats_indices = None
        self.matmul_2_cast = matmul_2_cast
        if pool_size is None:
            self.pool_size = None
        else:
            self.pool_size = numeric_tuple(pool_size, length=2)
        if window_size is None:
            self.window_size = None
            attention_size = input_size
        else:
            self.window_size = numeric_tuple(window_size, length=2)
            attention_size = self.window_size
            if relative_embedding_size is not None:
                relative_embedding_size = self.window_size
        self.scale = sqrt(dim // heads)

        # Set up submodules.
        self.input_layer_norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.qkv = CountedLinear(in_features=dim, out_features=dim * 3)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        if relative_embedding_size is not None:
            self.relative_position = RelativePositionEmbedding(
                attention_size,
                relative_embedding_size,
                dim // heads,
                pool_size=self.pool_size,
            )
        else:
            self.relative_position = None
        self.matmul = CountedMatmul()
        self.projection = CountedLinear(in_features=dim, out_features=dim)
        self.add = CountedAdd()
        self.mlp_layer_norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.mlp_1 = CountedLinear(in_features=dim, out_features=dim * mlp_ratio)
        self.gelu = nn.GELU()
        self.mlp_2 = CountedLinear(in_features=dim * mlp_ratio, out_features=dim)

    def forward(self, x):
        skip_1 = x
        x = self.input_layer_norm(x)

        # Linearly project x into qkv space.
        x = self.qkv(x)

        # Compute attention on the qkv representation.
        x, ats_indices = self._forward_attention(x)
        skip_1 = self._gather_ats_skip(skip_1, ats_indices)

        # Apply the post-attention linear transform and add the skip.
        x = self.projection(x)
        x = self.add(self.drop_path(x), skip_1)

        # Apply the token-wise MLP.
        skip_2 = x
        x = self.mlp_layer_norm(x)
        x = self._forward_mlp(x)
        x = self.add(self.drop_path(x), skip_2)
        return x

    def reset_self(self):
        self.last_ats_indices = None

    # A simple version of the method from
    # "Adaptive Token Sampling for Efficient Vision Transformers"
    # (Fayyaz et al., ECCV 2022)
    # For now we just use the top-k version of ATS (select the tokens
    # with the k highest scores). Using CDF-based token sampling should
    # also be possible, but it would be more complex to implement (we
    # would need a mechanism for masking the K' < K active tokens in
    # gates and buffers).
    def _adaptive_token_sampling(self, a, v):
        if self.ats_fraction is None:
            return a, None

        class_scores = a[..., 0]
        raw_scores = class_scores * torch.linalg.vector_norm(v[...], dim=-1)
        scores = raw_scores / raw_scores[..., 1:].sum(dim=-1, keepdim=True)

        # Always select the class token.
        scores[..., 0] = float("inf")

        # Sum scores over heads.
        scores = scores.sum(dim=-3)

        # Add +1 for the class token
        n_select = int(self.ats_fraction * (scores.shape[-1] - 1)) + 1

        # Select the k tokens with the highest scores.
        ats_indices = scores.topk(n_select, sorted=False)[1]

        # Sort the token indices (for stabilization). This seems to
        # work pretty well, although we could probably come up with
        # better/more sophisticated. E.g., we could try to find the
        # permutation of indices that minimized some norm between the
        # previous and current ats_indices.
        ats_indices = self._stabilize_ats_indices(ats_indices)
        self.last_ats_indices = ats_indices

        return (
            a.gather(dim=-2, index=expand_row_index(ats_indices, a.shape)),
            ats_indices,
        )

    def _cast_matmul_2(self, x, v):
        old_dtype = x.dtype
        if self.matmul_2_cast is not None:
            dtype = getattr(torch, self.matmul_2_cast)
            x = x.to(dtype)
            v = v.to(dtype)
        return x, v, old_dtype

    def _compute_window_padding(self):
        pad_h = -self.input_size[0] % self.window_size[0]
        pad_w = -self.input_size[1] % self.window_size[1]
        return pad_h, pad_w

    @staticmethod
    def _gather_ats_skip(skip_1, ats_indices):
        if ats_indices is None:
            return skip_1
        else:
            return skip_1.gather(
                dim=-2, index=expand_row_index(ats_indices, skip_1.shape)
            )

    def _forward_attention(self, x):
        # (batch, token, dim)

        # Partition the windows and attention heads. _window_partition
        # is a noop if self.window_size is None. Windows are arranged
        # along the batch dimension.
        x = self._partition_windows(x, in_qkv_domain=True)
        q, k, v = self._partition_heads(x)
        # (batch, heads, token, dim / heads)

        # Token pooling is a noop if self.pool_size is None.
        k = self._pool_tokens(k)
        v = self._pool_tokens(v)

        # Perform the actual attention computation.
        # The output of this first matmul is huge - hence it's much
        # faster to scale one of the inputs than it is to scale the
        # output.
        x = self.matmul(q / self.scale, k.transpose(-2, -1))
        if self.relative_position is not None:
            x = self.relative_position(x, q)
        x = x.softmax(dim=-1)

        # Adaptive token sampling is a noop if self.ats_fraction is None.
        x, ats_indices = self._adaptive_token_sampling(x, v)

        x, v, old_dtype = self._cast_matmul_2(x, v)
        x = self.matmul(x, v)
        # (batch, heads, token, dim / heads)

        x = self._recombine_heads(x)
        x = self._recombine_windows(x)
        x = self._uncast_matmul_2(x, old_dtype)
        # (batch, token, dim)

        return x, ats_indices

    def _forward_mlp(self, x):
        x = self.mlp_1(x)
        x = self.gelu(x)
        x = self.mlp_2(x)
        return x

    def _partition_heads(self, x):
        # (batch, token, dim)

        x = x.view(x.shape[:-1] + (3, self.heads, x.shape[-1] // (3 * self.heads)))
        q, k, v = x.permute(2, 0, 3, 1, 4)
        # (batch, heads, token, dim / heads)

        return q, k, v

    def _partition_windows(self, x, in_qkv_domain):
        if self.window_size is None:
            return x

        p = self._compute_window_padding()
        d = self.window_size
        # (batch, token, dim)

        # Unflatten the spatial dimensions.
        x = x.view(x.shape[:1] + self.input_size + x.shape[2:])
        # (batch, height, width, dim)

        if any(p):
            s = x.shape
            pad_tensor = torch.zeros(
                (1,) * (x.ndim - 1) + s[-1:], dtype=x.dtype, device=x.device
            )

            # The attention computation expects padded tokens to equal
            # _forward_qkv(zero). If x has already been mapped to the
            # QKV domain, then we need to transform the padded zero
            # values to the QKV domain. Only the bias portion of the
            # linear transform has an effect on the zero padding vector.
            if in_qkv_domain:
                pad_tensor = self.qkv.forward_bias(pad_tensor)

            # Pad to a multiple of the window size.
            # func.pad seems broken (see the comments in pad_to_size).
            # In the meantime we'll use pad_to_size.
            # x = func.pad(x, (0, 0, 0, p[1], 0, p[0]))
            x = pad_to_size(x, (s[-3] + p[0], s[-2] + p[1], s[-1]), pad_tensor)
            # (batch, height, width, dim)

        # Partition into windows.
        s = x.shape
        x = x.view(-1, s[-3] // d[0], d[0], s[-2] // d[1], d[1], s[-1])
        x = x.transpose(-3, -4)
        # (batch, window_y, window_x, token_y, token_x, dim)

        # Re-flatten the spatial dimensions. Can't use x.view here
        # because of the transpose.
        x = x.reshape(-1, prod(d), s[-1])
        # (batch * window, token, dim)

        return x

    def _pool_tokens(self, x):
        # (batch, heads, token, dim)

        if self.pool_size is None:
            return x
        w = self.input_size if (self.window_size is None) else self.window_size
        s = x.shape

        # Can't use x.view here because of the permutation in
        # _partition_heads.
        x = x.reshape((-1,) + w + x.shape[-1:])
        # (batch * heads, token_y, token_x, dim)

        x = x.permute(0, 3, 1, 2)
        x = func.avg_pool2d(x, self.pool_size)
        # (batch * heads, dim, token_y, token_x)

        x = x.permute(0, 2, 3, 1)
        # (batch * heads, token_y, token_x, dim)

        x = x.view(s[:-2] + (-1,) + s[-1:])
        # (batch, heads, token, dim)

        return x

    @staticmethod
    def _recombine_heads(x):
        # (batch, heads, token, dim / heads)

        # Can't use x.view here because of the permutation.
        x = x.permute(0, 2, 1, 3)
        x_reshaped = x.reshape(x.shape[:-2] + (-1,))
        # (batch, token, dim)

        # We assume that x.reshape actually copies the data. We can run
        # into problems if this is not the case, i.e., we may end up
        # with a gate being passed a raw reference to an accumulator
        # state. For an example, see EventfulMatmul1Block.
        assert x.data_ptr() != x_reshaped.data_ptr()
        x = x_reshaped

        return x

    def _recombine_windows(self, x):
        if self.window_size is None:
            return x

        p = self._compute_window_padding()
        d = self.window_size
        s = self.input_size
        total_h = p[0] + s[0]
        total_w = p[1] + s[1]
        # (batch * window, token, dim)

        # Unflatten the spatial dimensions.
        x = x.view(-1, total_h // d[0], total_w // d[1], d[0], d[1], x.shape[-1])
        # (batch, window_y, window_x, token_y, token_x, dim)

        # Recombine the window partitions. Can't use x.view here because
        # of the transpose.
        x = x.transpose(-3, -4)
        x = x.reshape(-1, total_h, total_w, x.shape[-1])
        # (batch, height, width, dim)

        # Remove padding.
        if any(p):
            x = x[:, : s[0], : s[1]]
            # (batch, height, width, dim)

        # Re-flatten the spatial dimensions.
        x = x.flatten(start_dim=1, end_dim=2)
        # (batch, token, dim)

        return x

    def _stabilize_ats_indices(self, ats_indices):
        ats_indices = ats_indices.sort(dim=-1)[0]
        if self.last_ats_indices is None:
            return ats_indices

        # Faster on the CPU
        new_indices = ats_indices.flatten(end_dim=-2).cpu()
        old_indices = self.last_ats_indices.flatten(end_dim=-2).cpu()
        stabilized = old_indices.clone()
        for i in range(new_indices.shape[0]):
            old_not_in_new = torch.isin(old_indices[i], new_indices[i], invert=True)
            new_not_in_old = torch.isin(new_indices[i], old_indices[i], invert=True)
            stabilized[i, old_not_in_new] = new_indices[i, new_not_in_old]
        return stabilized.to(ats_indices.device).view(ats_indices.shape)

    def _uncast_matmul_2(self, x, old_dtype):
        if self.matmul_2_cast is not None:
            x = x.to(old_dtype)
        return x


class EventfulTokenwiseBlock(Block):
    """
    A Transformer block that adds eventfulness to token-wise operations.
    """

    def __init__(self, gate_before_ln=False, stgt=False, **super_kwargs):
        """
        :param gate_before_ln: Determines whether token gates are placed
        before or after layer norm operations
        :param stgt: Whether to use the SimpleSTGTGate (instead of our
        TokenGate) for benchmarking
        :param super_kwargs: Kwargs for the super class (Block)
        """
        super().__init__(**super_kwargs)
        self.gate_before_ln = gate_before_ln
        token_gate_class = SimpleSTGTGate if stgt else TokenGate
        self.qkv_gate = token_gate_class()
        self.qkv_accumulator = TokenBuffer()
        self.projection_gate = token_gate_class()
        self.projection_accumulator = TokenBuffer()
        self.mlp_gate = token_gate_class()
        self.mlp_accumulator = TokenBuffer()

    def forward(self, x):
        skip_1, x, index = self._forward_pre_attention(x)
        x = self.qkv_accumulator(x, index)
        x, ats_indices = self._forward_attention(x)
        skip_1 = self._gather_ats_skip(skip_1, ats_indices)
        x = self._forward_post_attention(x, skip_1)
        return x

    def _forward_post_attention(self, x, skip_1):
        # Gate-accumulator block 2
        x, index = self.projection_gate(x)
        x = self.projection(x)
        x = self.projection_accumulator(x, index)

        x = self.add(self.drop_path(x), skip_1)
        skip_2 = x

        # Gate-accumulator block 3
        if self.gate_before_ln:
            x, index = self.mlp_gate(x)
            x = self.mlp_layer_norm(x)
        else:
            x = self.mlp_layer_norm(x)
            x, index = self.mlp_gate(x)
        x = self._forward_mlp(x)
        x = self.mlp_accumulator(x, index)
        x = self.add(self.drop_path(x), skip_2)

        return x

    def _forward_pre_attention(self, x):
        skip_1 = x

        # Gate-accumulator block 1
        if self.gate_before_ln:
            x, index = self.qkv_gate(x)
            x = self.input_layer_norm(x)
        else:
            x = self.input_layer_norm(x)
            x, index = self.qkv_gate(x)
        x = self.qkv(x)
        return skip_1, x, index


class EventfulMatmul1Block(EventfulTokenwiseBlock):
    """
    An EventfulTokenWiseBlock that adds eventfulness to the query-key
    product (in addition to token-wise operations).
    """

    def __init__(self, **super_kwargs):
        """
        :param super_kwargs: Kwargs for the super class (
        EventfulTokenwiseBlock)
        """
        super().__init__(**super_kwargs)

        # self._pool_index assumes that the input size is divisible by
        # the pooling size.
        if self.pool_size is not None:
            assert all(s % p == 0 for s, p in zip(self.input_size, self.pool_size))

        # This class only supports non-windowed attention for now.
        assert self.window_size is None

        self.matmul_accumulator_1 = MatmulBuffer()

    def forward(self, x):
        skip_1, x, index = self._forward_pre_attention(x)
        x = self.qkv_accumulator(x, index)
        x, ats_indices = self._forward_attention((x, index))
        skip_1 = self._gather_ats_skip(skip_1, ats_indices)
        x = self._forward_post_attention(x, skip_1)
        return x

    def _forward_attention(self, x):
        x, v, _ = self._forward_matmul_1(x)
        x, ats_indices = self._adaptive_token_sampling(x, v)
        x, v, old_dtype = self._cast_matmul_2(x, v)
        x = self.matmul(x, v)
        x = self._recombine_heads(x)
        x = self._uncast_matmul_2(x, old_dtype)
        return x, ats_indices

    def _forward_matmul_1(self, x):
        x, index = x
        q, k, v = self._partition_heads(x)
        k = self._pool_tokens(k)
        v = self._pool_tokens(v)
        index_k = self._pool_index(index)

        # See comment in Block._forward_attention.
        x = self.matmul_accumulator_1(
            q / self.scale, k.transpose(-2, -1), index, index_k
        )

        if self.relative_position is not None:
            # We need inplace=False because x is a direct reference to
            # an accumulator state.
            x = self.relative_position(x, q, inplace=False)
        x = x.softmax(dim=-1)
        return x, v, index_k

    def _pool_index(self, index):
        if (self.pool_size is None) or (index is None):
            return index
        width = self.input_size[1]
        index_y = index.div(width, rounding_mode="floor")
        index_x = index.remainder(width)
        index_y = index_y.div(self.pool_size[0], rounding_mode="floor")
        index_x = index_x.div(self.pool_size[1], rounding_mode="floor")
        index = index_y * (width // self.pool_size[1]) + index_x

        # Calling .unique() still works if there are multiple items in
        # the batch. However, the output size along dim=-1 will be the
        # largest of the individual output sizes. This could result in
        # some redundant downstream computation.
        index = index.unique(dim=-1)
        return index


class EventfulBlock(EventfulMatmul1Block):
    """
    An EventfulMatmul1Block that also adds eventfulness to the
    attention-value product.
    """
    def __init__(self, **super_kwargs):
        """
        :param super_kwargs: Kwargs for the super class (
        EventfulTokenwiseBlock)
        """
        super().__init__(**super_kwargs)
        self.v_gate = TokenDeltaGate()
        self.matmul_gate = TokenDeltaGate(structure="col")
        self.matmul_accumulator_2 = MatmulDeltaAccumulator()

    def _forward_attention(self, a):
        a, v, index_k = self._forward_matmul_1(a)

        a, v, old_dtype = self._cast_matmul_2(a, v)
        a, ats_indices = self._adaptive_token_sampling(a, v)
        if not self.matmul_2_cast:
            # We clone v here because it may be a direct reference to
            # self.qkv_accumulator.a.
            v = v.clone()
        v_n_tilde, v_delta_tilde, index_v = self.v_gate(v, forced_index=index_k)
        a_n_tilde, a_delta_tilde, _ = self.matmul_gate(a, forced_index=index_v)
        a = self.matmul_accumulator_2(
            a_n_tilde, v_n_tilde, a_delta_tilde, v_delta_tilde
        )

        a = self._recombine_heads(a)
        a = self._uncast_matmul_2(a, old_dtype)
        return a, ats_indices
