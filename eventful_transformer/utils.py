import torch
from math import prod
from torch import nn as nn
from torch.nn import functional as func

from eventful_transformer.base import ExtendedModule
from eventful_transformer.counting import CountedAdd, CountedEinsum


class DropPath(ExtendedModule):
    """
    Defines a drop-path module.

    Reference: https://github.com/alibaba-mmai-research/TAdaConv/blob/main/models/base/base_blocks.py
    """
    def __init__(self, drop_rate):
        """
        :param drop_rate: Fraction that should be dropped
        """
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep_mask = torch.rand(shape, device=x.device) > self.drop_rate
        output = x.div(1.0 - self.drop_rate) * keep_mask.to(x.dtype)
        return output


class PositionEncoding(ExtendedModule):
    """
    Defines a position encoding module.
    """
    def __init__(self, dim, encoding_size, input_size, has_class_token):
        """
        :param dim: The dimensionality of token vectors
        :param encoding_size: The size (in tokens) assumed for position
        encodings
        :param input_size: The expected size of the inputs in tokens
        :param has_class_token: Whether the input has a class token
        """
        super().__init__()
        self.encoding_size = tuple(encoding_size)
        self.input_size = tuple(input_size)
        self.has_class_token = has_class_token
        tokens = prod(self.encoding_size) + int(has_class_token)
        self.encoding = nn.Parameter(torch.zeros(1, tokens, dim))
        self.add = CountedAdd()
        self.cached_encoding = None

    def forward(self, x):
        if self.training:
            self.cached_encoding = None
            encoding = self._compute_sized_encoding()
        else:
            # Cache the resized encoding during inference (assuming the
            # weights don't change, its value doesn't change between
            # model invocations).
            if self.cached_encoding is None:
                self.cached_encoding = self._compute_sized_encoding()
            encoding = self.cached_encoding

        # Add the position encoding.
        x = self.add(x, encoding)
        return x

    def _compute_sized_encoding(self):
        encoding = self.encoding

        # Interpolate the position encoding if needed.
        if self.input_size != self.encoding_size:
            # (batch, patch, dim)

            if self.has_class_token:
                # The class token comes *first* (see ViViTSubModel).
                class_token = encoding[:, :1]
                encoding = encoding[:, 1:]
            else:
                class_token = None
            encoding = encoding.transpose(1, 2)
            encoding = encoding.view(encoding.shape[:-1] + self.encoding_size)
            # (batch, dim) + encoding_size

            # Note: We do not count operations from this interpolation,
            # even though it is in the backbone. This is because the
            # cost of interpolating is amortized over many invocations.
            encoding = func.interpolate(
                encoding, self.input_size, mode="bicubic", align_corners=False
            )
            # (batch, dim) + embedding_size

            encoding = encoding.flatten(start_dim=2)
            encoding = encoding.transpose(1, 2)
            if self.has_class_token:
                encoding = torch.concat([class_token, encoding], dim=1)
            # (batch, patch, dim)

        return torch.Tensor(encoding)

    def reset_self(self):
        # Clear the cached value of sized_encoding whenever the model is
        # reset (just in case new weights get loaded).
        self.cached_encoding = None


class RelativePositionEmbedding(ExtendedModule):
    """
    Defines relative position embeddings.
    """
    def __init__(self, attention_size, embedding_size, head_dim, pool_size=None):
        """
        :param attention_size: The expected size of the attention window
        :param embedding_size: The size (in tokens) assumed for position
        embeddings
        :param head_dim: The dimensionality of each attention head
        :param pool_size: The pooling size (if self-attention pooling is
        being used - see the pool_size parameter to Block.
        """
        super().__init__()
        self.attention_size = attention_size
        self.embedding_size = embedding_size
        self.pool_size = pool_size
        self.y_embedding = nn.Parameter(
            torch.zeros(2 * embedding_size[0] - 1, head_dim)
        )
        self.x_embedding = nn.Parameter(
            torch.zeros(2 * embedding_size[1] - 1, head_dim)
        )
        self.add = CountedAdd()
        self.einsum = CountedEinsum()
        self.y_relative = None
        self.x_relative = None

    # This is based on the add_decomposed_rel_pos function here:
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/utils.py
    # noinspection PyTypeChecker
    def forward(self, x, q, inplace=True):
        a = self.attention_size

        # Unflatten the spatial dimensions.
        if self.pool_size is None:
            p = a
        else:
            p = (a[0] // self.pool_size[0], a[1] // self.pool_size[1])
        x = x.view(x.shape[:2] + a + p)
        q = q.view(q.shape[:2] + a + q.shape[-1:])

        # Apply the relative position embedding.
        if self.y_relative is None:
            # Cache y_relative and x_relative (assuming the weights
            # don't change, their values don't change between model
            # invocations).
            self.y_relative = self._get_relative(self.y_embedding, dim=0)
            self.x_relative = self._get_relative(self.x_embedding, dim=1)
        x = self.add(
            x,
            self.einsum("abhwc,hkc->abhwk", q, self.y_relative).unsqueeze(dim=-1),
            inplace=inplace,
        )
        x = self.add(
            x,
            self.einsum("abhwc,wkc->abhwk", q, self.x_relative).unsqueeze(dim=-2),
            inplace=True,
        )

        # Re-flatten the spatial dimensions.
        x = x.view(x.shape[:2] + (prod(a), prod(p)))

        return x

    # This is a simplification of the get_rel_pos function here:
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/utils.py
    def _get_relative(self, embedding, dim):
        range_0 = torch.arange(self.embedding_size[dim]).unsqueeze(dim=1)
        range_1 = torch.arange(self.embedding_size[dim]).unsqueeze(dim=0)
        relative = embedding[range_0 - range_1 + self.embedding_size[dim] - 1]
        if self.embedding_size != self.attention_size:
            relative = relative.transpose(0, 2).unsqueeze(dim=0)
            relative = func.interpolate(
                relative, self.attention_size, mode="bicubic", align_corners=False
            )
            relative = relative.squeeze(dim=0).transpose(0, 2)
        if self.pool_size is not None:
            relative = relative.transpose(1, 2)
            relative = func.avg_pool1d(relative, self.pool_size[dim])
            relative = relative.transpose(1, 2)
        return relative

    def reset_self(self):
        # Clear the cached values of x_relative and y_relative whenever
        # the model is reset (just in case new weights get loaded).
        self.y_relative = None
        self.x_relative = None


def expand_col_index(index, target_shape):
    old_shape = index.shape
    new_dims = len(target_shape) - index.ndim
    index = index.view(old_shape[:-1] + (1,) * new_dims + old_shape[-1:])
    index = index.expand(target_shape[:-1] + (-1,))
    return index


def expand_row_index(index, target_shape):
    old_shape = index.shape
    new_dims = len(target_shape) - index.ndim
    index = index.view(old_shape[:-1] + (1,) * (new_dims - 1) + (old_shape[-1], 1))
    index = index.expand(target_shape[:-2] + (-1, target_shape[-1]))
    return index
