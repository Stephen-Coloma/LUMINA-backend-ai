import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.model.nsclc_model import NSCLC_Model
from src.utils.logger import setup_logger
from src.utils.yaml_loader import load_model_config as yml_load
from src.utils.dataset import MedicalDataset
from pathlib import Path

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

def train(model, train_loader, val_loader, criterion, optimizer, config, device, logger):
    num_epochs = config['training']['epochs']

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (label, ct, pet) in enumerate(train_loader):
            label, ct, pet = label.to(device), ct.to(device), pet.to(device)

            optimizer.zero_grad()

            # forward pass
            output = model(ct, pet)
            loss = criterion(output, label)

            # backward pass and optimizer
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_training_loss = running_loss / len(train_loader)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] | "
                    f"Training Loss: {avg_training_loss:.4f} | "
                    f"Validation Loss: {val_loss:.4f} | "
                    f"Validation Accuracy: {val_acc:.2f}%")

        # save model checkpoint
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

def main():
    dataset_path = Path(r'D:\Datasets\Output')
    logger = setup_logger(Path('../logs'), 'Training.log', 'TrainingLogger')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = yml_load('configs/model.yml')

    model = NSCLC_Model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # load dataset
    dataset = MedicalDataset(dataset_path)

    # split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # create dataloader instances
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['validation']['batch_size'], shuffle=False)

    train(model, train_loader, val_loader, criterion, optimizer, config, device, logger)

if __name__ == '__main__':
    main()






