import torch
import torch.nn as nn
from src.model.classification_block import ClassificationBlock
from src.model.feature_extraction_block import FeatureExtractionBlock
from src.model.fusion_block import FusionBlock

class NSCLC_Model(nn.Module):
    def __init__(self, model_config):
        super(NSCLC_Model, self).__init__()

        # obtain parameters from the model configuration
        self.num_classes = model_config['model']['num_classes']

        # feature block parameters
        in_channels = model_config['feature_block']['in_channels']
        growth_rate = model_config['feature_block']['growth_rate']
        num_layers = model_config['feature_block']['num_layers']

        # model architecture
        self.feature_extraction_ct = FeatureExtractionBlock(in_channels, growth_rate, num_layers)
        self.feature_extraction_pet = FeatureExtractionBlock(in_channels, growth_rate, num_layers)

        # fusion block
        self.fusion_block = FusionBlock()

        # classification block
        fusion_channels = 2 * (in_channels + num_layers * growth_rate)
        self.classification_block = ClassificationBlock(in_channels=fusion_channels, num_classes=self.num_classes)

    def forward(self, x_ct, x_pet):
        # extract features
        x_ct_features = self.feature_extraction_ct(x_ct)
        x_pet_features = self.feature_extraction_pet(x_pet)

        # fusion through concatenation
        x_fused = self.fusion_block(x_ct_features, x_pet_features)

        # classification
        x = self.classification_block(x_fused)

        return x