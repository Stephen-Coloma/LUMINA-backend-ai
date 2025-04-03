import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== FEATURE BLOCK =====================
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
        return F.relu(out, inplace=True)

class DenseBlock(nn.Module):
    """ Dense Block:
            Sequence of layers where each layer extracts new features and concatenates its output with all previous
            feature maps enabling feature reuse.

        Architecture:
            Dense Block: Conv3D(3x3x3) -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, growth_rate, num_layers):
        """
        Args:
        :param in_channels: Number of input channels before entering the Dense Block.
        :param growth_rate: Number of output feature maps generated per layer.
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
                nn.BatchNorm3d(growth_rate),
                nn.ReLU(inplace=True)
            ))

        # apply CBAM after feature extraction
        # self.cbam = CBAM(in_channels + num_layers * growth_rate) TODO: just a placeholder, fix it after CBAM is implemented

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x) # obtain new feature maps from the current layer

            # concatenate new feature maps with the previous ones
            x = torch.cat([x, new_features], dim=1)

        # x = self.cbam(x) TODO: just a placeholder, fix it after CBAM is implemented
        return x


# ===================== CONVOLUTIONAL BLOCK ATTENTION MODULE (CBAM) =====================
# TODO: create 3 classes (channel attention, spatial attention, and CBAM to integrate the channel and spatial attention to be used for the dense block)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.shape

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        out = out.view(b, c, 1, 1, 1)
        return x * out
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm3d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
        
class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x