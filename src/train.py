import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.model.nsclc_model import NSCLC_Model
from src.utils.logger import setup_logger
from src.utils.yaml_loader import load_model_config as yml_load
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

def load_dataset(dataset_path: Path, logger) -> List[Dict[str, Any]]:
    patient_list = []

    for data_file in dataset_path.glob('*.npy'):
        try:
            data = np.load(data_file, allow_pickle=True).item()
            patient_list.append(data)
        except Exception as e:
            logger.warning(f'Failed to load {data_file.name}: {e}')
    return patient_list

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for ct, pet, labels in val_loader:
            ct, pet, labels = ct.to(device), pet.to(device), labels.to(device)
            outputs = model(ct, pet)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy

def train(model, train_loader, val_loader, criterion, optimizer, config, device, logger):
    num_epochs = config['training']['epochs']

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (ct, pet, labels) in enumerate(train_loader):
            ct, pet, labels = ct.to(device), pet.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(ct, pet)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_training_loss = running_loss / len(train_loader)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] | "
                    f"Training Loss: {avg_training_loss:.4f} | "
                    f"Validation Loss: {val_loss:.4f} | "
                    f"Validation Accuracy: {val_acc:.2f}%")

def main():
    dataset_path = Path(r'D:\Datasets\Output')
    logger = setup_logger(Path('../logs'), 'Training.log', 'TrainingLogger')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = yml_load('configs/model.yml')

    model = NSCLC_Model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Load dataset
    dataset = load_dataset(dataset_path, logger)

    # TODO: Replace with a proper Dataset class for torch DataLoader
    # This dummy split assumes dataset elements are (ct, pet, label) tuples
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    train(model, train_loader, val_loader, criterion, optimizer, config, device, logger)


if __name__ == '__main__':
    main()






