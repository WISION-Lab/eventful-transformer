import torch
import torch.nn as nn
from torchvision.transforms import Normalize

from eventful_transformer.backbones import ViTBackbone
from eventful_transformer.base import ExtendedModule
from eventful_transformer.blocks import LN_EPS
from eventful_transformer.counting import CountedLinear
from utils.image import as_float32, resize_to_fit


# Resources consulted:
# https://github.com/google-research/scenic/tree/main/scenic/projects/vivit
# https://github.com/alibaba-mmai-research/TAdaConv
# https://github.com/alibaba-mmai-research/TAdaConv/blob/main/models/base/transformer.py


# See configs/models/vivit_b_kinetics400.yml for an example
# configuration.
class FactorizedViViT(ExtendedModule):
    def __init__(
        self,
        classes,
        input_shape,
        normalize_mean,
        normalize_std,
        spatial_config,
        spatial_views,
        temporal_config,
        temporal_stride,
        temporal_views,
        tubelet_shape,
        batch_views=True,
        dropout_rate=0.0,
        spatial_only=False,
        temporal_only=False,
    ):
        super().__init__()
        assert not (spatial_only and temporal_only)
        assert not (dropout_rate < 0.0 or dropout_rate > 1.0)
        input_shape = tuple(input_shape)
        tubelet_shape = tuple(tubelet_shape)
        input_t, input_c, input_h, input_w = input_shape
        backbone_input_size = (input_h // tubelet_shape[1], input_w // tubelet_shape[2])
        self.batch_views = batch_views
        self.spatial_only = spatial_only
        self.temporal_only = temporal_only

        # Set up submodules.
        self.preprocessing = ViViTPreprocessing(
            input_shape,
            normalize_mean,
            normalize_std,
            temporal_stride,
            temporal_views,
            spatial_views,
        )
        dim = spatial_config["block_config"]["dim"]
        self.embedding = TubeletEmbedding(input_c, dim, tubelet_shape)
        self.spatial_model = ViViTSubModel(backbone_input_size, spatial_config)
        backbone_input_size = (input_t // tubelet_shape[0],)
        self.temporal_model = ViViTSubModel(backbone_input_size, temporal_config)
        self.dropout = (
            nn.Dropout(dropout_rate) if (dropout_rate > 0.0) else nn.Identity()
        )
        self.classifier = CountedLinear(in_features=dim, out_features=classes)

    def forward(self, x):
        batch_size = x.shape[0]
        if not self.temporal_only:
            x = self._forward_spatial(x)
        if not self.spatial_only:
            x = self._forward_temporal(x, batch_size)
        return x

    def _forward_spatial(self, x):
        # Performance note: We can improve performance somewhat by
        # batching multiple views together and calling _forward_view
        # once. However, this complicates things when we select
        # different numbers of tokens from different views (e.g., with
        # a threshold policy) and want to perform some operation (e.g.,
        # a matrix multiply) where the batch dimension needs to be
        # maintained. We may want to revisit this.
        # Idea: We could restrict the number of active tokens to a small
        # number of preset values (e.g., 0, 32, 64, 128) and group items
        # in the batch by the number of active tokens. This would reduce
        # the number of kernel invocations - from the number of items in
        # the batch to the number of preset values (4 in the previous
        # example).
        x = self.preprocessing(x)
        if self.batch_views:
            x = torch.stack(x, dim=1).flatten(end_dim=1)
            x = self._forward_view(x)
        else:
            x = [self._forward_view(view) for view in x]
            x = torch.stack(x, dim=1).flatten(end_dim=1)
        return x

    def _forward_temporal(self, x, batch_size):
        x = x.view((-1,) + x.shape[-2:])
        x = self.temporal_model(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = x.view(batch_size, -1, x.shape[-1])
        x = x.mean(dim=-2)
        x = x.softmax(dim=-1)
        return x

    def _forward_view(self, x):
        # (batch, time, channel, height, width)

        x = self.embedding(x)
        # (batch, time, patch, dim)

        # Apply the spatial model to each time step.
        self.spatial_model.reset()
        x = torch.stack([self.spatial_model(x[:, t]) for t in range(x.shape[1])], dim=1)
        # (batch, time, dim)

        return x


class TubeletEmbedding(nn.Module):
    def __init__(self, input_channels, dim, tubelet_shape):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=input_channels,
            out_channels=dim,
            kernel_size=tubelet_shape,
            stride=tubelet_shape,
        )

    def forward(self, x):
        # (batch, time, dim, height, width)

        # Permute so all 3 dimensions for Conv3d are adjacent.
        x = x.permute(0, 2, 1, 3, 4)

        x = self.conv(x)
        # (batch, dim, time, height, width)

        # Flatten the spatial axes.
        x = x.flatten(start_dim=-2)
        # (batch, dim, time, patch)

        x = x.permute(0, 2, 3, 1)
        # (batch, time, patch, dim)

        return x


class ViViTPreprocessing(nn.Module):
    def __init__(
        self,
        input_shape,
        normalize_mean,
        normalize_std,
        temporal_stride,
        temporal_views,
        spatial_views,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.temporal_stride = temporal_stride
        self.temporal_views = temporal_views
        self.spatial_views = spatial_views
        self.normalization = Normalize(normalize_mean, normalize_std)

    def forward(self, x):
        t, _, h, w = self.input_shape

        # Repeat the last frame if the video is too short.
        view_size = self.temporal_stride * t
        if x.shape[1] < view_size:
            n_pad = view_size - x.shape[1]
            frame = x[:, -1:]
            pad_frames = frame.expand(frame.shape[:1] + (n_pad,) + frame.shape[2:])
            x = torch.concat([x, pad_frames], dim=1)

        # Chop the video into multiple temporal views.
        if self.temporal_views == 1:
            start_positions = [(x.shape[1] - view_size) // 2]
        else:
            spacing = (x.shape[1] - view_size) / (self.temporal_views - 1)
            start_positions = [int(k * spacing) for k in range(self.temporal_views)]
        x = [x[:, i : i + view_size : self.temporal_stride] for i in start_positions]

        # Normalize and resize the video.
        x = [as_float32(x_i) for x_i in x]  # Range [0, 1]
        x = [self.normalization(x_i) for x_i in x]
        x = [
            torch.stack(
                [resize_to_fit(x_i[:, t], (h, w)) for t in range(x_i.shape[1])], dim=1
            )
            for x_i in x
        ]

        # Chop each temporal view into multiple spatial views.
        if self.spatial_views == 1:
            start_positions = [((x[0].shape[-2] - h) // 2, (x[0].shape[-1] - w) // 2)]
        else:
            h_spacing = (x[0].shape[-2] - h) / (self.spatial_views - 1)
            w_spacing = (x[0].shape[-1] - w) / (self.spatial_views - 1)
            start_positions = [
                (int(k * h_spacing), int(k * w_spacing))
                for k in range(self.spatial_views)
            ]
        x = [view[..., i : i + h, j : j + w] for i, j in start_positions for view in x]

        return x


class ViViTSubModel(ExtendedModule):
    def __init__(self, input_size, backbone_config):
        super().__init__()
        dim = backbone_config["block_config"]["dim"]
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.backbone = ViTBackbone(
            input_size=input_size, has_class_token=True, **backbone_config
        )
        self.layer_norm = nn.LayerNorm(dim, eps=LN_EPS)

    def forward(self, x):
        # Append the class token.
        expand_shape = (x.shape[0],) + self.class_token.shape[1:]
        x = torch.concat([self.class_token.expand(expand_shape), x], dim=1)

        x = self.backbone(x)
        x = self.layer_norm(x)

        # Extract the class embedding token.
        x = x[:, 0]
        return x
