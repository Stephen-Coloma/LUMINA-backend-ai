# ==== Torch Imports ====
import torch.nn as nn
import torch.optim as optim


def get_optimizer(name, **kwargs):
    """
    Returns a PyTorch optimizer based on the given name.

    Args:
    :param name: The name of the optimizer to be retrieved.
    :param kwargs: Additional arguments to pass to the optimizer.
    :return: An instance of the requested optimizer.
    """
    if name == 'adagrad':
        return optim.Adagrad(**kwargs)
    elif name == 'rmsprop':
        return optim.RMSprop(**kwargs)
    elif name == 'adam':
        return optim.Adam(**kwargs)
    elif name == 'adamw':
        return optim.AdamW(**kwargs)

def get_loss_fn(name, **kwargs):
    """
    Returns a PyTorch loss function based on the given name.

    Args:
    :param name: The name of the loss function to be retrieved.
    :param kwargs: Additional arguments to pass to the loss function.
    :return: An instance of the requested loss function.
    """
    if name == 'crossentropyloss':
        return nn.CrossEntropyLoss(**kwargs)