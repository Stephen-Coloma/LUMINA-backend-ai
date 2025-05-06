import torch.nn as nn
from src.model.ai.classification_block import ClassificationBlock
from src.model.ai.feature_extraction_block import FeatureExtractionBlock
from src.model.ai.fusion_block import FusionBlock

class NSCLC_Model(nn.Module):
    def __init__(self, model_config):
        super(NSCLC_Model, self).__init__()

        # obtain parameters from the model configuration
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

        # print(f'x_ct_features shape: {x_ct_features.shape}')
        # print(f'x_pet_features shape: {x_pet_features.shape}')

        # fusion through concatenation
        x_fused = self.fusion_block(x_ct_features, x_pet_features)

        # print(f'fused_features: {x_fused.shape}')

        # classification
        x = self.classification_block(x_fused)

        # print(f'classification output: {x}')

        return x