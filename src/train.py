import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model.nsclc_model import NSCLC_Model
from src.utils.yaml_loader import load_model_config as yml_load

# define device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
config = yml_load('configs/model.yml')
model = NSCLC_Model(config).to(device)

# define loss function
criterion = nn.CrossEntropyLoss()

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

# TODO: data transformation if applicable

# TODO: load dataset
# training_dataset =
# train_loader = DataLoader(training_dataset, batch_size=4, shuffle=True)
#
# # training loop
# num_epochs = config['training']['epochs']
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#
#     for i, (ct, pet, labels) in enumerate(train_loader):
#         ct, pet, labels = ct.to(device), pet.to(device), labels.to(device)
#
#         # zero gradients
#         optimizer.zero_grad()
#
#         # forward pass
#         outputs = model(ct, pet)
#
#         # loss computation
#         loss = criterion(outputs, labels)
#
#         # backward pass and optimization
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')