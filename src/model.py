# ==== Third Party Imports ====
import torch.nn as nn

# ==== Local Project Imports ====
from blocks import FeatureExtractionBlock, FusionBlock, ClassificationBlock


# ========== DenseNet 3D Model Class ==========
class DenseNet3D(nn.Module):
    def __init__(self, model_config):
        super(DenseNet3D, self).__init__()

        # obtain parameters from the blocks configuration
        self.num_classes = model_config.data.num_classes

        # feature block parameters
        in_channels = model_config.feature_block.in_channels
        growth_rate = model_config.feature_block.growth_rate
        use_transition = model_config.feature_block.use_transition
        compression = model_config.feature_block.compression
        num_blocks = model_config.feature_block.dense_block.blocks
        num_layers = model_config.feature_block.dense_block.layers

        # feature extraction block
        self.feature_extraction_ct = FeatureExtractionBlock(
            in_channels=in_channels,
            growth_rate=growth_rate,
            num_blocks=num_blocks,
            num_layers=num_layers,
            use_transition=use_transition,
            compression=compression
        )
        self.feature_extraction_pet = FeatureExtractionBlock(
            in_channels=in_channels,
            growth_rate=growth_rate,
            num_blocks=num_blocks,
            num_layers=num_layers,
            use_transition=use_transition,
            compression=compression
        )

        # output channels for ct and pet after feature extraction
        ct_output_channels = self.feature_extraction_ct.out_channels
        pet_output_channels = self.feature_extraction_pet.out_channels

        # fusion block
        fusion_channels = int((ct_output_channels + pet_output_channels) / 2)
        self.fusion_block = FusionBlock(fusion_channels)

        # classification block
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