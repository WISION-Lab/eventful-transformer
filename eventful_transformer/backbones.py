import torch.nn as nn

from eventful_transformer import blocks
from eventful_transformer.base import ExtendedModule
from eventful_transformer.utils import PositionEncoding


class ViTBackbone(ExtendedModule):
    """
    Common backbone for vision Transformers.
    """

    def __init__(
        self,
        block_config,
        depth,
        position_encoding_size,
        input_size,
        block_class="Block",
        has_class_token=False,
        window_indices=(),
        windowed_class=None,
        windowed_overrides=None,
    ):
        """
        :param block_config: A dict containing kwargs for the
        block_class constructor
        :param depth: The number of blocks to use
        :param position_encoding_size: The size (in tokens) assumed for
        position encodings
        :param input_size: The expected size of the inputs in tokens
        :param block_class: The specific Block class to use (see
        blocks.py for options)
        :param has_class_token: Whether to add an extra class token
        :param window_indices: Block indices that should use windowed
        attention
        :param windowed_class: The specific Block class to use with
        windowed attention (if None, fall back to block_class)
        :param windowed_overrides: A dict containing kwargs overrides
        for windowed_class
        """
        super().__init__()
        self.position_encoding = PositionEncoding(
            block_config["dim"], position_encoding_size, input_size, has_class_token
        )
        self.blocks = nn.Sequential()
        for i in range(depth):
            block_class_i = block_class
            block_config_i = block_config.copy()
            if i in window_indices:
                if windowed_class is not None:
                    block_class_i = windowed_class
                if windowed_overrides is not None:
                    block_config_i |= windowed_overrides
            else:
                block_config_i["window_size"] = None
            self.blocks.append(
                getattr(blocks, block_class_i)(input_size=input_size, **block_config_i)
            )

    def forward(self, x):
        x = self.position_encoding(x)
        x = self.blocks(x)
        return x
