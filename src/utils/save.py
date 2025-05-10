# ==== Standard
from typing import Union
from pathlib import Path

# ==== PyTorch Import ====
import torch


def save_checkpoint(epoch, model, optimizer, losses, path):
    """
    Saves a checkpoint of the current model. A dictionary containing
    the current epoch, model, optimizer, and losses will be saved.

    Args:
    :param epoch: The current epoch under training.
    :param model: The PyTorch model to be saved as a checkpoint.
    :param optimizer: The optimizer used by the model during training.
    :param losses: The current loss values.
    :param path: The output path where the checkpoint will be saved.
    :return: None
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses,
    }, path)

def save_model(model, path):
    """
    Saves a PyTorch model using TorchScript tracing.

    Args:
    :param model: The PyTorch model to be saved.
    :param path: The output path where the model will be saved.
    :return: None
    """
    model.eval()
    example_input = torch.randn(1, 1, 32, 200, 200)

    traced_model = torch.jit.trace(model, (example_input, example_input))
    traced_model.save(path)
