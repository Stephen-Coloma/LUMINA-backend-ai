import torch
import torch.nn as nn

class FusionBlock(nn.Module):
    def __init__(self, in_channels):
        super(FusionBlock, self).__init__()

    def forward(self, x_ct, x_pet):
        return x_ct + x_pet