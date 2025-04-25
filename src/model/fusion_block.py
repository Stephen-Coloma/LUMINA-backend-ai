import torch
import torch.nn as nn

class FusionBlock(nn.Module):
    """ Fusion Block
            Concatenates feature maps of two different modalities along the channel dimension
            to create a unified representation for downstream processing.
    """
    def __init__(self):
        super(FusionBlock, self).__init__()

    def forward(self, feature_ct, feature_pet):
        """
        Args:
        :param feature_ct: Feature map of the CT modality
        :param feature_pet: Feature map of the PET modality
        :return:
        """
        fused = torch.cat([feature_ct, feature_pet], dim=1)
        return fused