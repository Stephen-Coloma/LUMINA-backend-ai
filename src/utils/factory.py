import torch.nn as nn
import torch.optim as optim

def get_optimizer(name, **kwargs):
    if name == 'adagrad':
        return optim.Adagrad(**kwargs)
    elif name == 'rmsprop':
        return optim.RMSprop(**kwargs)
    elif name == 'adam':
        return optim.Adam(**kwargs)
    elif name == 'adamw':
        return optim.AdamW(**kwargs)

def get_loss_fn(name, **kwargs):
    if name == 'crossentropyloss':
        return nn.CrossEntropyLoss(**kwargs)