import torch
import torch.nn as nn
import torch.nn.functional as func
from math import prod

from eventful_transformer.base import ExtendedModule, numeric_tuple


class CountedAdd(ExtendedModule):
    """
    An addition operator that counts flops.
    """

    def forward(self, a, b, inplace=False):
        if inplace:
            a += b
            result = a
        else:
            result = a + b
        if self.count_mode:
            self.counts["add_flops"] += result.numel()
        return result


class CountedBias(ExtendedModule):
    """
    A bias-addition module that counts flops.
    """

    def __init__(self, features, spatial_dims=0, device=None, dtype=None):
        """
        :param features: Dimensionality of the bias (size of feature
        dimension)
        :param spatial_dims: The number of trailing spatial dimensions
        of the input
        :param device: Bias device
        :param dtype: Bias data type
        """
        super().__init__()
        self.features = features
        self.spatial_dims = spatial_dims
        self.bias = nn.Parameter(torch.zeros(features, device=device, dtype=dtype))

    def forward(self, x):
        result = x + self.bias.view((self.features,) + (1,) * self.spatial_dims)
        if self.count_mode:
            self.counts["bias_flops"] += result.numel()
        return result


class CountedConv(ExtendedModule):
    """
    A convolution module that counts flops.
    """

    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        device=None,
        dtype=None,
    ):
        """
        :param spatial_dims: The number of spatial dims (e.g., 2 for 2D
        convolution)
        :param in_channels: The number of input channels
        :param out_channels: The number of output channels
        :param kernel_size: The kernel size (int or tuple)
        :param stride: The convolution stride (int or tuple)
        :param padding: The amount of padding
        :param dilation: Dilation ratio
        :param groups: Number of channel groups
        :param device: Convolution kernel device
        :param dtype: Convolution kernel data type
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = numeric_tuple(kernel_size, length=spatial_dims)
        self.stride = numeric_tuple(stride, length=spatial_dims)
        if isinstance(padding, int):
            self.padding = numeric_tuple(padding, length=spatial_dims)
        else:
            self.padding = padding
        self.dilation = numeric_tuple(dilation, length=spatial_dims)
        self.groups = groups
        self.conv_function = getattr(func, f"conv{self.spatial_dims}d")
        shape = (out_channels, in_channels // groups) + self.kernel_size
        self.weight = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))

    def forward(self, x):
        result = self.conv_function(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.count_mode:
            fan_in = (self.in_channels // self.groups) * prod(self.kernel_size)
            self.counts[f"conv{self.spatial_dims}d_flops"] += result.numel() * fan_in
        return result


class CountedEinsum(ExtendedModule):
    """
    Einsum (Einstein summation) operation that counts flops.
    """

    def forward(self, equation, *operands):
        if self.count_mode:
            # There might be some cases here I haven't considered. But
            # this works fine for inner products.
            op_map = torch.einsum(equation, *[torch.ones_like(x) for x in operands])
            self.counts["einsum_flops"] += int(op_map.sum())
        return torch.einsum(equation, *operands)


class CountedLinear(ExtendedModule):
    """
    Linear transform operation that counts flops.
    """

    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        :param in_features: Dimensionality of input vectors
        :param out_features: Dimensionality of output vectors
        :param device: Transform matrix device
        :param dtype: Transform matrix data type
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        shape = (out_features, in_features)
        self.weight = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))

    def forward_bias(self, x):
        result = x + self.bias
        if self.count_mode:
            self.counts["bias_flops"] += result.numel()
        return result

    def forward_linear(self, x):
        if self.count_mode:
            self.counts["linear_flops"] += x.numel() * self.out_features
        return func.linear(x, self.weight)

    def forward(self, x):
        result = func.linear(x, self.weight, self.bias)
        if self.count_mode:
            self.counts["bias_flops"] += result.numel()
            self.counts["linear_flops"] += x.numel() * self.out_features
        return result


class CountedMatmul(ExtendedModule):
    """
    Matrix multiplication operation that counts flops. We assume a
    batched 2D matrix multiplication.
    """

    def forward(self, a, b):
        result = a @ b
        if self.count_mode:
            self.counts["matmul_flops"] += result.numel() * a.shape[-1]
        return result
