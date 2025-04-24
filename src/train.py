import torch
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

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for label, ct, pet in val_loader:
            label, ct, pet = label.to(device), ct.to(device), pet.to(device)
            outputs = model(ct, pet)
            loss = criterion(outputs, label)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == label).sum().item()
            total_samples += label.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy

def train(model, train_loader, val_loader, loss_fn, optimizer, config, device, logger):
    model.train()
    num_epochs = config.training.epochs
    running_loss = 0.0

    start_timer()
    for epoch in range(num_epochs):
        try:
            logger.info(f'Starting epoch {epoch + 1}')

            for target, ct, pet in train_loader:
                target, ct, pet = target.to(device), ct.to(device), pet.to(device)

                optimizer.zero_grad()

                # forward pass with mixed precision
                with autocast(device.type):
                    output = model(ct, pet)
                    loss = loss_fn(output, target)

                # backward pass
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            val_loss, val_acc = validate(model, val_loader, loss_fn, device)

            logger.info(
                f'Epoch [{epoch + 1}/{num_epochs}] | '
                f'Training Loss: {avg_train_loss:.4f} | '
                f'Validation Loss: {val_loss:.4f} | '
                f'Validation Accuracy: {val_acc:.2f}%'
            )

            # save model checkpoint
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            torch.save(model.state_dict(), f'../model/checkpoints/{timestamp}.pth')

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

    # train model
    logger.info('Training Started')
    train(model, train_loader, val_loader, loss_fn, optimizer, config, device, logger)

if __name__ == '__main__':
    main()






