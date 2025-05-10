import torch
from torch import nn as nn
from torch.nn import functional as f
from app.models.ai.cbam import CBAM

# ===================== Feature Extraction Block =====================
class FeatureExtractionBlock(nn.Module):
    """ Feature Extraction Block
            Combines Dense Block, CBAM, and an optional Transition Layer, repeated per block.

        Architecture:
            Input -> Dense Block -> CBAM -> Transition Layer -> Output
    """
    def __init__(self, in_channels, growth_rate, num_blocks, num_layers, use_transition=True, compression=0.5):
        """
        Args:
        :param in_channels: Number of input channels.
        :param growth_rate: Number of new feature maps added to the output by each layer in the Dense Block.
                            (i.e. growth in channel dimension per layer)
        :param num_layers: Number of layers in the Dense Block.
        :param use_transition: Activation of transition layer
        :param compression: Factor to reduce the number of channels
        """
        super(FeatureExtractionBlock, self).__init__()
        self.blocks = nn.ModuleList()
        self.use_transition = use_transition
        self.compression = compression

        current_channels = in_channels

        for i in range(num_blocks):
            # Dense Block
            dense_block = DenseBlock(current_channels, growth_rate, num_layers)
            self.blocks.append(dense_block)
            current_channels += num_layers * growth_rate

            # CBAM
            self.blocks.append(CBAM(current_channels))

            # Transition
            if use_transition and i != 1:
                reduced_channels = int(current_channels * compression)
                transition = Transition(current_channels, reduced_channels)
                self.blocks.append(transition)
                current_channels = reduced_channels

        self.out_channels = current_channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# ===================== Dense Block =====================
class DenseBlock(nn.Module):
    """ Dense Block:
            Sequence of layers where each layer extracts new features and concatenates its output with all previous
            feature maps enabling feature reuse.

        Architecture:
            [Bottleneck -> Conv3D(3x3x3) -> InstanceNorm -> ReLU] x num_layers
    """
    def __init__(self, in_channels, growth_rate, num_layers):
        """
        Args:
        :param in_channels: Number of input channels before entering the Dense Block.
        :param growth_rate: Number of new feature maps added to the output by each layer.
                            (i.e. growth in channel dimension per layer)
        :param num_layers: Number of layers in the Dense Block.
        """
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                # apply a bottleneck layer to reduce feature maps
                Bottleneck(in_channels + i * growth_rate, growth_rate),

                # apply a 3x3x3 convolution to extract new features
                # in_channel is also equal to growth_rate due to the bottleneck layer
                nn.Conv3d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm3d(growth_rate, affine=True),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x) # obtain new feature maps from the current layer

            # concatenate new feature maps with the previous ones
            x = torch.cat([x, new_features], dim=1)
        return x

# ===================== Bottleneck Layer =====================
class Bottleneck(nn.Module):
    """ Bottleneck Layer:
            Applies a 1x1x1 convolution to reduce the number of input feature maps before passing them to the
            dense block. This enforces the output feature maps to be equal to 'growth_rate', controlling feature
            expansion.

        Architecture:
            Conv3D(1x1x1) -> InstanceNorm -> ReLU
    """
    def __init__(self, in_channels, growth_rate):
        """
        Args:
        :param in_channels: Number of input channels before the bottleneck layer.
        :param growth_rate:  Number of output feature maps after the bottleneck layer.
        """
        super(Bottleneck, self).__init__()
        self.conv = nn.Conv3d(in_channels, growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn = nn.InstanceNorm3d(growth_rate, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# ===================== Transition Layer =====================
class Transition(nn.Module):
    """ Transition Layer
            Reduces the number of feature maps and spatial dimensions.

        Architecture:
            Conv3D(1x1x1) -> InstanceNorm -> ReLU -> AvgPool3D(2x2x2)
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
        :param in_channels: The number of input feature channels.
        :param out_channels: The number of output feature channels.
        """
        super(Transition, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        return x