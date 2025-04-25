import torch
import torch.nn as nn

class ClassificationBlock(nn.Module):
    """ Classification Block:
            Applies global average pooling and a fully connected layer to generate class scores.
            Dropout is applied before the final classification layer to reduce overfitting.

        Architecture:
            GlobalAvgPool -> Dropout -> Fully Connected Layer
    """
    def __init__(self, in_channels, num_classes, dropout_val=0.5):
        """
        Args:
        :param in_channels: Number of input feature maps.
        :param num_classes: Number of output classes.
        :param dropout_val: Dropout probability before the classification layer.
        """
        super(ClassificationBlock, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_val)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
