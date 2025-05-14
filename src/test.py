# ==== Third Party Imports ====
import torch
from torch import nn as nn

# ==== Local Project Imports ====
from src.model import DenseNet3D
from src.utils.config_loader import load_config
from src.utils.load_dataset import get_dataset, get_test_dataset
from src.utils.validate_or_test_model import validate_or_test


def main():
    # config file
    config = load_config('../configs/model.yml')

    # device type
    device = torch.device('cuda')

    # load dataset
    dataset = get_dataset(config)
    test_dl = get_test_dataset(dataset, config)

    # load model
    model = torch.jit.load('../model/best_model_20250509_211950.pt')
    model = model.to(device)
    model.eval()

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # metrics computation
    results = validate_or_test(model, test_dl, loss_fn, device, 'Testing')
    avg_loss, accuracy, precision, recall, f1, conf_mtx = results

    print(
        f'\nTEST RESULTS:\n'
        f'> Avg Loss: {avg_loss:.4f}\n'
        f'> Accuracy: {accuracy:.4f}\n'
        f'> Precision: {precision:.4f}\n'
        f'> Recall: {recall:.4f}\n'
        f'> F1 Score: {f1:.4f}\n'
        'Confusion Matrix:\n'
        f'{conf_mtx}'
    )

if __name__ == '__main__':
    main()



