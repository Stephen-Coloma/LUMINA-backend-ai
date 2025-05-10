# ==== Third Party Imports ====
import torch
from torch import nn


class CBAM(nn.Module):
    """
    Enhances feature representation by applying both
    channel and spatial attention mechanisms.

    Architecture:
        Input -> ChannelAttention -> Multiply -> SpatialAttention -> Multiple -> Output
    """

    def __init__(self, in_channels):
        """
        Initializes CBAM object.

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


class ChannelAttention(nn.Module):
    """
    Emphasizes important channels and suppressing less
    relevant ones.

    Architecture:
        Input -> AvgPool3d -> Shared MLP -> +
            -> MaxPool3d -> SharedMLP -> +
            -> Sigmoid -> Scale
    """

    def __init__(self, in_channels, reduction_ratio=16):
        """
        Initializes ChannelAttention object.

        Args:
        :param in_channels: Number of input channels.
        :param reduction_ratio: Reduction ratio for the internal hidden layer in the MLP.
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """
    Emphasizes informative regions in space by learning where
    to focus within feature maps.

    Architecture:
        Input -> AvgPool(channel) + MaxPool(channel) -> Concat -> Conv3d(kernel_size) -> Sigmoid -> Scale
    """

    def __init__(self, kernel_size=3):
        """
        Initializes SpatialAttention object.

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