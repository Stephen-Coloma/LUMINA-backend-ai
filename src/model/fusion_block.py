import torch
import torch.nn as nn

class FusionBlock(nn.Module):
    def __init__(self):
        super(FusionBlock, self).__init__()

    def forward(self, feature_ct, feature_pet):
        fused = torch.cat([feature_ct, feature_pet], dim=1)
        return fused