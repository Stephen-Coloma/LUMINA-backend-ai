# ==== Third Party Imports ====
import torch
import torch.nn as nn


class ClassificationBlock(nn.Module):
    """
    Performs the final classification by aggregating
    spatial features and applying a fully connected layer.

    Architecture:
        AdaptivePool3d -> Flatten -> Dropout -> Linear Layer
    """

    def __init__(self, in_channels, num_classes):
        """
        Initializes ClassificationBlock object.

        Args:
        :param in_channels: Number of input feature maps.
        :param num_classes: Number of output classes for classification.
        """
        super(ClassificationBlock, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
