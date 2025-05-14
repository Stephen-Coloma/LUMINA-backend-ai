# ==== Third Party Imports ====
import torch
from torch import nn as nn

# ==== Local Project Imports ====
from src.blocks.cbam import CBAM


class FeatureExtractionBlock(nn.Module):
    """
    The feature extraction block is composed of multiple Dense
    Blocks. Each block may include an attention mechanism and
    a transition layer for spatial/channel downsampling.

    Architecture:
        Input -> Dense Block * num_blocks -> Output
    """

    def __init__(self, in_channels, growth_rate, num_blocks, num_layers, use_cbam=True, use_transition=True, compression=0.5):
        """
        Initializes FeatureExtractionBlock object.

        Args:
        :param in_channels: Number of input channels.
        :param growth_rate: Number of feature maps each DenseBlock layer adds.
        :param num_blocks: Number of Dense Block sequences.
        :param num_layers: Number of layers in each Dense Block.
        :param use_cbam: Flag indicating if an attention mechanism will be utilized.
        :param use_transition: Flag indicating if a transition layer will be utilized.
        :param compression: Factor to reduce the number of channels in the transition layer.
        """
        super(FeatureExtractionBlock, self).__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for i in range(num_blocks):
            block = DenseBlock(
                in_channels=current_channels,
                growth_rate=growth_rate,
                num_layers=num_layers,
                use_cbam=use_cbam,
                use_transition=(use_transition and i == 0), # only add transition after the first block if use_transition is true
                compression=compression
            )
            self.blocks.append(block)
            current_channels = block.out_channels

        self.out_channels = current_channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DenseBlock(nn.Module):
    """
    A Dense Block for 3D volumetric feature extraction.

    Architecture (per layer):
        [Bottleneck -> Conv3d(3x3x3) -> InstanceNorm3d -> ReLU] * num_layers -> CBAM -> Transition Layer
    """

    def __init__(self, in_channels, growth_rate, num_layers, use_cbam=True, use_transition=True, compression=0.5):
        """
        Initializes DenseBlock object.

        Args:
        :param in_channels: Input feature channels.
        :param growth_rate: Output channels added per layer.
        :param num_layers: Number of dense layers.
        :param use_cbam: Flag indicating if an attention mechanism will be utilized.
        :param use_transition: Flag indicating if a transition layer will be utilized.
        :param compression: Factor to reduce the number of channels in the transition layer.
        """
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.use_cbam = use_cbam
        self.use_transition = use_transition

        current_channels = in_channels

        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                # apply a bottleneck layer to reduce feature maps
                Bottleneck(in_channels + i * growth_rate, growth_rate),

                # apply a 3x3x3 convolution to extract new features
                nn.Conv3d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm3d(growth_rate, affine=True),
                nn.ReLU(inplace=True),
                # nn.Dropout3d(p=0.1) Not Used BTW
            ))
            current_channels += growth_rate

        if self.use_cbam:
            self.attn = nn.Sequential(
                CBAM(current_channels),
                nn.Dropout3d(p=0.1)
            )

        if self.use_transition:
            reduced_channels = int(current_channels * compression)
            self.transition = Transition(current_channels, reduced_channels)
            self.out_channels = reduced_channels
        else:
            self.out_channels = current_channels

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x) # obtain new feature maps from the current layer

            # concatenate new feature maps with the previous ones
            x = torch.cat([x, new_features], dim=1)

        if self.use_cbam:
            x = self.attn(x)

        if self.use_transition:
            x = self.transition(x)

        return x


class Bottleneck(nn.Module):
    """
    A bottleneck layer that reduces input feature dimensionality
    before a 3D convolution.

    Architecture:
        Conv3d(1x1x1) -> InstanceNorm3d -> ReLU
    """

    def __init__(self, in_channels, growth_rate):
        """
        Args:
        :param in_channels: Input feature channels.
        :param growth_rate:  Output channels after the bottleneck layer.
        """
        super(Bottleneck, self).__init__()
        self.conv = nn.Conv3d(in_channels, growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm = nn.InstanceNorm3d(growth_rate, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Transition(nn.Module):
    """
    A transition layer to reduce the number of channels
    and spatial resolution.

    Architecture:
        Conv3d(1x1x1) -> InstanceNorm3d -> ReLU -> AvgPool3d(1x2x2)
    """

    def __init__(self, in_channels, out_channels):
        """
        Initializes Transition object.

        Args:
        :param in_channels: Input feature channels.
        :param out_channels: Output feature channels after compression.
        """
        super(Transition, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        return x