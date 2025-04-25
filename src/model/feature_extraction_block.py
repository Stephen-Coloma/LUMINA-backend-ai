import torch
from torch import nn as nn
from torch.nn import functional as f
from src.model.cbam import CBAM

# ===================== Bottleneck Layer =====================
class Bottleneck(nn.Module):
    """ Bottleneck Layer:
            Applies a 1x1x1 convolution to reduce the number of input feature maps before passing them to the
            dense block. This enforces the output feature maps to be equal to 'growth_rate', controlling feature
            expansion.

        Architecture:
            Conv3D(1x1x1) -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, growth_rate):
        """
        Args:
        :param in_channels: Number of input channels before the bottleneck layer.
        :param growth_rate:  Number of output feature maps after the bottleneck layer.
        """
        super(Bottleneck, self).__init__()
        self.conv = nn.Conv3d(in_channels, growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm3d(growth_rate)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return f.relu(out, inplace=True)

# ===================== Dense Block =====================
class DenseBlock(nn.Module):
    """ Dense Block:
            Sequence of layers where each layer extracts new features and concatenates its output with all previous
            feature maps enabling feature reuse.

        Architecture:
            [Bottleneck -> Conv3D(3x3x3) -> BatchNorm -> ReLU -> Dropout] x num_layers
    """
    def __init__(self, in_channels, growth_rate, num_layers, dropout_val=0.2):
        """
        Args:
        :param in_channels: Number of input channels before entering the Dense Block.
        :param growth_rate: Number of new feature maps added to the output by each layer.
                            (i.e. growth in channel dimension per layer)
        :param num_layers: Number of layers in the Dense Block.
        :param dropout_val: Dropout probability applied after each Dense Block layer.
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
                nn.BatchNorm3d(growth_rate),
                nn.ReLU(inplace=True),

                # apply dropout
                nn.Dropout3d(p=dropout_val)
            ))

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x) # obtain new feature maps from the current layer

            # concatenate new feature maps with the previous ones
            x = torch.cat([x, new_features], dim=1)
        return x

# ===================== Feature Extraction Block =====================
class FeatureExtractionBlock(nn.Module):
    """ Feature Extraction Block
           Combines a Dense Block and CBAM for feature extraction and attention. A dropout layer is
           applied at the end of the block for regularization.

        Architecture:
            Input -> Dense Block -> CBAM -> Dropout -> Output
    """
    def __init__(self, in_channels, growth_rate, num_layers, dropout_val=0.3):
        """
        Args:
        :param in_channels: Number of input channels.
        :param growth_rate: Number of new feature maps added to the output by each layer in the Dense Block.
                            (i.e. growth in channel dimension per layer)
        :param num_layers: Number of layers in the Dense Block.
        :param dropout_val: Dropout probability applied after CBAM.
        """
        super(FeatureExtractionBlock, self).__init__()
        self.dense_block = DenseBlock(in_channels, growth_rate, num_layers)
        self.cbam = CBAM(in_channels + num_layers * growth_rate)
        self.dropout = nn.Dropout3d(p=dropout_val)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.cbam(x)
        return x
