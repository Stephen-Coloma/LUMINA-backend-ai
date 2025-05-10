# ==== Third Party Imports ====
import torch
import torch.nn as nn


class FusionBlock(nn.Module):
    """
    A feature fusion block that concatenates feature maps
    of two different modalities along the channel dimension
    and applies additional processing.

    Architecture:
        [InstanceNorm3d CT, InstanceNorm3d PET] -> Concat -> Conv3d(3x3x3) -> ReLU
    """

    def __init__(self, num_features):
        """
        Initializes FusionBlock object.

        :param num_features: The number of input channels for each modality
        """
        super(FusionBlock, self).__init__()
        self.norm_ct = nn.InstanceNorm3d(num_features, affine=True)
        self.norm_pet = nn.InstanceNorm3d(num_features, affine=True)
        self.conv = nn.Conv3d(2 * num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, feature_ct, feature_pet):
        # normalize both modalities using InstanceNorm
        feature_ct = self.norm_ct(feature_ct)
        feature_pet = self.norm_pet(feature_pet)

        # concatenate feature maps along the channel dimension
        fused = torch.cat([feature_ct, feature_pet], dim=1)

        # process fused feature map further with a convolutional layer
        fused = self.conv(fused)
        fused = self.relu(fused)

        return fused