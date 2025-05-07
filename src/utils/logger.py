import logging
from pathlib import Path

def setup_logger(log_path: Path, log_filename: str, logger_name: str) -> logging.Logger:
    log_file = log_path / log_filename
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

def log_configuration(config, logger, device_name):
    logger.info(
        '\nGPU:\n'
        f'> Model: {device_name}'
        'TRAINING:\n'
        f'> Epochs: {config.training.epochs}\n'
        f'> Batch Size: {config.training.batch_size} (Training) | {config.validation.batch_size} (Validation)\n'
        'EARLY STOP:\n'
        f'> Patience: {config.training.early_stop.patience}'
        f'> Min Delta: {config.training.early_stop.min_delta}'
        'OPTIMIZER:\n'
        f'> Name: {config.optimizer.name}'
        f'> Learning Rate: {config.optimizer.lr}\n'
        f'> Weight Decay: {config.optimizer.weight_decay}\n'
        'LOSS FUNCTION:\n'
        f'> Name: {config.loss.name}\n'
        f'> Label Smoothing: {config.loss.label_smoothing}'
        'FEATURE BLOCK:\n'
        f'> Growth Rate: {config.feature_block.growth_rate}\n'
        f'> Use Transition: {config.feature_block.use_transition}\n'
        f'> Compression: {config.feature_block.compression}\n'
        'DENSE BLOCK:\n'
        f'> Num Blocks: {config.feature_block.dense_block.blocks}\n'
        f'> Num Layers: {config.feature_block.dense_block.layers}'
    )