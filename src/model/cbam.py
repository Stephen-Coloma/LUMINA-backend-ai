import torch
from torch import nn

# ===================== Channel Attention Module =====================
class ChannelAttention(nn.Module):
    """ Channel Attention Module
            Learns to emphasize informative feature channels and suppress less useful ones.
            It aggregates spatial information using both average and max pooling, applied after is a
            shared multi-layer perceptron to compute attention weights for each channel.

        Architecture:
            [AvgPool3D -> MPL -> Sigmoid] + [MaxPool3D -> MLP -> Sigmoid] -> Add -> Scale
    """
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Args:
        :param in_channels: Number of input channels.
        :param reduction_ratio: Reduction ratio for the internal hidden layer in the MLP.
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
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


# ===================== Spatial Attention Module =====================
class SpatialAttention(nn.Module):
    """ Spatial Attention Module
            Learns to focus on important spatial locations.
            It uses average and max pooling along the channel axis and applies a convolution
            to compute for spatial attention weights.

        Architecture:
            [AvgPool along Channel] + [MaxPool along channel] -> Concat -> Conv3D(3x3x3) -> Sigmoid
    """
    def __init__(self, kernel_size=3):
        """
        Args:
        :param kernel_size: Kernel size for the convolutional layer (must be odd).
        """
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


# ===================== Convolutional Block Attention Module (CBAM) =====================
class CBAM(nn.Module):
    """ Convolutional Block Attention Module (CBAM)
            Sequentially applies channel attention and spatial attention to support feature extraction by
            enhancing important features in both dimensions.

        Architecture:
            Input -> ChannelAttention -> Multiply -> SpatialAttention -> Multiply -> Output
    """
    def __init__(self, in_channels):
        """
        Args:
        :param in_channels: Number of input channels to apply attention on.
        """
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
