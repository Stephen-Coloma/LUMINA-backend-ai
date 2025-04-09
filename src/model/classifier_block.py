import torch
import torch.nn as nn

class ClassifierBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ClassifierBlock, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  

        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):

        x = self.global_avg_pool(x)  

        x = torch.flatten(x, 1)      
        x = self.fc(x)               

        return x
