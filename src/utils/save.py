import torch

def save_densenet(epoch, model, optimizer, losses, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses,
    }, path)

def save_naivebayes():
    # TODO: create a function to save the ml model
    return