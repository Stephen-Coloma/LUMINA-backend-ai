import torch
from torch.cpu.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from src.model.nsclc_model import NSCLC_Model
from src.utils.logger import setup_logger
from src.utils.dataset import MedicalDataset
from src.utils.config_loader import load_config
from src.utils.timing import start_timer, end_timer_and_print
from pathlib import Path
from torch.amp import autocast
from src.utils.factory import get_optimizer, get_loss_fn
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def validate(model, val_loader, criterion, device, logger):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for labels, ct, pet in val_loader:
            # label, ct_inputs, pet_inputs = label.to(device), ct.to(device), pet.to(device)
            outputs = model(ct, pet)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels().cpu().numpy)

            # loss = criterion(outputs, labels)
            #
            # total_loss += loss.item()
            # _, predicted = torch.max(outputs, 1)
            # total_correct += (predicted == labels).sum().item()
            # total_samples += labels.size(0)

    # avg_loss = total_loss / len(val_loader)
    # accuracy = 100.0 * total_correct / total_samples
    # return avg_loss, accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_mtx = confusion_matrix(all_labels, all_preds)

    logger.info(f'Val Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1:.2f}')
    logger.info(f'Confusion Matrix:\n{conf_mtx}')

def train(model, train_loader, val_loader, loss_fn, optimizer, config, device, logger, start_epoch = None):
    start_epoch = start_epoch or 0
    num_epochs = config.training.epochs

    scaler = GradScaler()

    start_timer()
    for epoch in range(start_epoch, num_epochs):
        try:
            model.train()
            running_loss = 0.0
            logger.info(f'Starting epoch {epoch + 1}')

            for target, ct, pet in train_loader:
                target, ct, pet = target.to(device), ct.to(device), pet.to(device)

                optimizer.zero_grad()

                # forward pass with mixed precision
                with autocast(device.type):
                    output = model(ct, pet)
                    loss = loss_fn(output, target)

                # backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            logger.info(
                f'Epoch [{epoch + 1}/{num_epochs}] | '
                f'Training Loss: {avg_train_loss:.4f}'
            )

            validate(model, val_loader, loss_fn, device, logger)

            # save model checkpoint
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'../model/checkpoints/{timestamp}.pth')

            allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            logger.info(f'GPU Memory - Allocated: {allocated:.2f}MB | Reserved: {reserved:.2f}MB')
            logger.info(f'Epoch {epoch + 1} ended')
        except Exception as e:
            logger.error(e)

    logger.info(f'Training Ended')
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
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # create dataloader instances
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        pin_memory=config.training.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
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
    train(model, train_loader, val_loader, loss_fn, optimizer, config, device, logger, start_epoch)

if __name__ == '__main__':
    main()






