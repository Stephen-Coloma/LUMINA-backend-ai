import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from src.model.nsclc_model import NSCLC_Model
from src.utils.logger import setup_logger
from src.utils.dataset import MedicalDataset
from src.utils.config_loader import load_config
from src.utils.timing import start_timer, end_timer_and_print
from pathlib import Path
from src.utils.factory import get_optimizer, get_loss_fn
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def validate(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for targets, ct_batch, pet_batch in data_loader:
            targets = targets.to(device, non_blocking=True)
            ct_batch = ct_batch.to(device, non_blocking=True)
            pet_batch = pet_batch.to(device, non_blocking=True)

            outputs = model(ct_batch, pet_batch)
            losses = loss_fn(outputs, targets)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            running_loss += losses.item()

    return compute_metrics(all_targets, all_preds, running_loss, len(data_loader))

def compute_metrics(all_targets, all_preds, running_loss, data_len):
    avg_loss = running_loss / data_len
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    conf_mtx = confusion_matrix(all_targets, all_preds)

    return (avg_loss, accuracy, precision, recall, f1, conf_mtx)

def train(model, train_loader, val_loader, loss_fn, optimizer, config, device, logger, start_epoch = None):
    start_epoch = start_epoch or 0
    num_epochs = config.training.epochs + 1

    scaler = GradScaler(device.type)

    start_timer()
    for epoch in range(start_epoch, num_epochs):
        try:
            model.train()
            logger.info(f'Starting epoch [{epoch + 1}/{num_epochs}]')

            all_preds = []
            all_targets = []

            # training phase
            optimizer.zero_grad()
            running_loss = 0.0

            for targets, ct_batch, pet_batch in train_loader:
                targets = targets.to(device, non_blocking=True)
                ct_batch = ct_batch.to(device, non_blocking=True)
                pet_batch = pet_batch.to(device, non_blocking=True)

                # forward pass with mixed precision
                with autocast(device.type):
                    outputs = model(ct_batch, pet_batch)
                    losses = loss_fn(outputs, targets)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

                # backward pass
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += losses.item()

            # metric computation for training data
            train_results = compute_metrics(all_targets, all_preds, running_loss, len(train_loader))

            # metric computation for validation data
            val_results = validate(model, val_loader, loss_fn, device)

            # store results in a log file
            log_metrics(train_results, logger, is_training=True)
            log_metrics(val_results, logger, is_training=False)

            # save model checkpoint
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses,
            }, f'../model/checkpoints/{timestamp}.pth')

            logger.info(f'Epoch {epoch + 1} ended')
        except Exception as e:
            logger.error(e)

    logger.info('Training Ended')
    end_timer_and_print('Default Precision:')

def main():
    # external model config file
    config = load_config('../configs/model.yml')

    # console and file logger
    logger = setup_logger(Path('../logs'), 'Training.log', 'TrainingLogger')

    # device type (cuda for nvidia gpu, else cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model instantiation
    model = NSCLC_Model(config).to(device)

    # optimizer
    optimizer = get_optimizer(
        name=config.optimizer.name,
        params=model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay
    )

    # loss
    loss_fn = get_loss_fn(
        name=config.loss.name
    )

    # load dataset
    dataset_path = Path(config.data.path)
    dataset = MedicalDataset(dataset_path)

    # split dataset into train and validation
    split_value = config.data.train_val_split
    train_size = int(split_value * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])

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

    # train model
    logger.info('Training Started')
    train(model, train_data_loader, validation_data_loader, loss_fn, optimizer, config, device, logger, start_epoch)

def log_metrics(results, logger, is_training=True):
    avg_loss, accuracy, precision, recall, f1_score, conf_mtx = results

    logger.info('\nTraining Results') if is_training else logger.info('\nValidation Results')
    logger.info(
        f'Avg Loss: {avg_loss:.4f}'
        f'Accuracy: {accuracy:.2f}\n'
        f'Precision: {precision:.2f}\n'
        f'Recall: {recall:.2f}\n'
        f'F1 Score: {f1_score:.2f}\n'
        'Confusion Matrix:\n'
        f'{conf_mtx}'
    )

if __name__ == '__main__':
    main()






