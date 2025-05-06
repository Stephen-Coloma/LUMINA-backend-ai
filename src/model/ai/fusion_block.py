import torch
import torch.nn as nn

class FusionBlock(nn.Module):
    """ Fusion Block
            Concatenates feature maps of two different modalities along the channel dimension
            and applies additional processing (normalization and convolution).
    """
    def __init__(self, num_features):
        super(FusionBlock, self).__init__()
        self.bn_ct = nn.InstanceNorm3d(num_features, affine=True)
        self.bn_pet = nn.InstanceNorm3d(num_features, affine=True)
        self.conv = nn.Conv3d(2 * num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, feature_ct, feature_pet):
        """
        Args:
        :param feature_ct: Feature map of the CT modality
        :param feature_pet: Feature map of the PET modality
        :return:
        """
        # normalize both modalities using InstanceNorm
        feature_ct = self.bn_ct(feature_ct)
        feature_pet = self.bn_pet(feature_pet)

        # concatenate feature maps along the channel dimension
        fused = torch.cat([feature_ct, feature_pet], dim=1)

        # process fused feature map further with a convolutional layer
        fused = self.conv(fused)
        fused = self.relu(fused)

        return fused