import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.model.nsclc_model import NSCLC_Model
from src.utils.yaml_loader import load_model_config as yml_load

# define device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
config = yml_load('configs/model.yml')

model = NSCLC_Model(config).to(device)

# define loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

# TODO: Load dataset (must be cleaned and preprocessed)
dataset =   

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


train_loader = DataLoader(training_dataset, batch_size=4, shuffle=True)

# Validation function for the model
def validate(model, val_loader, criterion):
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

# training loop
num_epochs = config['training']['epochs']

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (ct, pet, labels) in enumerate(train_loader):
        ct, pet, labels = ct.to(device), pet.to(device), labels.to(device)

        # zero gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(ct, pet)

        # loss computation
        loss = criterion(outputs, labels)

        # backward pass + optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_training_loss = running_loss / len(train_loader)
    validation_loss, validation_accuracy = validate(model, val_loader, criterion)

    print(f"Epoch [{epoch + 1}/{num_epochs}] | "
          f"Training Loss: {avg_training_loss:.4f} | "
          f"Validation Loss: {validation_loss:.4f} | "
          f"Validation Accuracy: {validation_accuracy:.2f}%")
