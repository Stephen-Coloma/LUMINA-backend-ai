# ==== Standard Imports ====
from pathlib import Path
from datetime import datetime

# ==== Third Party Imports ====
import torch
from torch import bfloat16
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# ==== Local Project Imports ====
from .model import DenseNet3D
from .data import MedicalDataset
from utils import load_config
from utils import EarlyStopping
from utils import get_optimizer, get_loss_fn
from utils import setup_logger, log_configuration
from utils import compute as compute_metrics, log as log_metrics
from utils import save_checkpoint, save_model


# ========== Train Function ==========
def train(model, train_loader, val_loader, loss_fn, optimizer, config, device, logger, start_epoch = None):
    torch.cuda.empty_cache()

    start_epoch = start_epoch or 0
    num_epochs = config.training.epochs + 1

    scaler = GradScaler(enabled=(device.type == 'cuda'), init_scale=64, growth_interval=128)

    # early stopping
    early_stopper = EarlyStopping(
        patience=config.training.early_stop.patience,
        min_delta=config.training.early_stop.min_delta
    )

    for epoch in range(start_epoch, num_epochs):
        try:
            logger.info(f'Starting epoch [{epoch + 1}/{num_epochs - 1}]')
            model.train()

            all_preds = []
            all_targets = []

            # training phase
            running_loss = 0.0
            total_samples = 0

            progress_bar = tqdm(train_loader, desc='Training', leave=False)
            for targets, ct_batch, pet_batch in progress_bar:
                optimizer.zero_grad()

                targets = targets.to(device, non_blocking=True)
                ct_batch = ct_batch.to(device, non_blocking=True)
                pet_batch = pet_batch.to(device, non_blocking=True)

                # forward pass with mixed precision
                with autocast(device.type, dtype=bfloat16):
                    outputs = model(ct_batch, pet_batch)
                    losses = loss_fn(outputs, targets)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                if torch.isnan(losses) or torch.isinf(losses):
                    logger.error("Loss is NaN or Inf. Skipping this batch.")
                    continue

                # backward pass
                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                batch_size = targets.size(0)
                running_loss += losses.item() * batch_size
                total_samples += batch_size
                progress_bar.set_postfix(loss=losses.item())

            # metric computation for training data
            train_results = compute_metrics(all_targets, all_preds, running_loss, total_samples)

            # metric computation for validation data
            val_results = validate(model, val_loader, loss_fn, device)

            # store results in a log file
            log_metrics(train_results, logger, is_training=True)
            log_metrics(val_results, logger, is_training=False)

            # determine if the training needs to stop early
            early_stopper(val_results[0])

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = Path(r'../model')

            if early_stopper.early_stop:
                logger.info(f'Early stopping triggered at epoch {epoch + 1}')
                save_model(model, (model_path / f'best_model_{timestamp}.pth'))
                early_stopper.reset()

            save_checkpoint(epoch, model, optimizer, losses, (model_path / 'checkpoints' / f'{timestamp}.pth'))

            logger.info(f'Epoch {epoch + 1} ended')

            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(e)

    logger.info('Training Ended')

# ========== Main Function ==========
def main():
    # external blocks config file
    config = load_config('../configs/model.yml')

    # console and file logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logger(Path('../logs'), f'{timestamp}_Training.log', 'TrainingLogger')

    # device type (cuda for nvidia gpu, else cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # blocks instantiation
    model = DenseNet3D(config).to(device)

    # optimizer
    optimizer = get_optimizer(
        name=config.optimizer.name,
        params=model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay
    )

    # loss function
    loss_fn = get_loss_fn(
        name=config.loss.name,
        label_smoothing=config.loss.label_smoothing
    )

    # load dataset
    dataset_path = Path(config.data.path)
    dataset = MedicalDataset(dataset_path)

    # extract labels
    labels = [dataset[i][0] for i in range(len(dataset))]

    # stratified shuffle split
    split_value = config.data.train_val_split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 - split_value,random_state=42)
    train_idx, val_idx = next(splitter.split(X=labels, y=labels))

    train_dataset = Subset(dataset, train_idx)
    validation_dataset = Subset(dataset, val_idx)

    # create dataloader instances
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        pin_memory=config.training.pin_memory
    )
    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=config.validation.batch_size,
        shuffle=config.validation.shuffle,
        pin_memory=config.validation.pin_memory
    )

    # load latest checkpoint if applicable
    checkpoint_files = sorted(Path(r'../model/checkpoints').glob('*.pth'), reverse=True)
    start_epoch = 0
    if checkpoint_files:
        last_checkpoint = checkpoint_files[0]
        checkpoint = torch.load(last_checkpoint, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Loaded checkpoint from {last_checkpoint.name} (epoch {checkpoint['epoch']})")
        torch.cuda.empty_cache()

    # log gpu and config setup
    log_configuration(config, logger, torch.cuda.get_device_name())

    # train blocks
    logger.info('Training Started')
    train(model, train_data_loader, validation_data_loader, loss_fn, optimizer, config, device, logger, start_epoch)

def validate(model, data_loader, loss_fn, device):
    model.eval()

    all_preds = []
    all_targets = []

    running_loss = 0.0
    total_samples = 0

    # with mixed precision
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Validating', leave=False)
        for targets, ct_batch, pet_batch in progress_bar:
            targets = targets.to(device, non_blocking=True)
            ct_batch = ct_batch.to(device, non_blocking=True)
            pet_batch = pet_batch.to(device, non_blocking=True)

            with autocast(device.type, dtype=bfloat16):
                outputs = model(ct_batch, pet_batch)
                losses = loss_fn(outputs, targets)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            batch_size = targets.size(0)
            running_loss += losses.item() * batch_size
            total_samples += batch_size
            progress_bar.set_postfix(loss=losses.item())

    torch.cuda.empty_cache()

    return compute_metrics(all_targets, all_preds, running_loss, total_samples)

# ========== Runnable ==========
if __name__ == '__main__':
    main()