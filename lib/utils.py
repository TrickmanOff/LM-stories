import torch

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def get_device():
    return DEVICE


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
