import torch.nn as nn
import torch.optim as optim

def get_optimizer(name, params, lr=1e-3, weight_decay=0):
    if name == 'sgd':
        return optim.SGD()
    elif name == 'adagrad':
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif name == 'adadelta':
        return optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)

def get_loss_fn(name, **kwargs):
    if name == 'crossentropyloss':
        return nn.CrossEntropyLoss(**kwargs)