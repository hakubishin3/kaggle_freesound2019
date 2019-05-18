import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Source: https://gist.github.com/AdrienLE/bf31dfe94569319f6e47b2de8df13416#file-focal_dice_1-py
class FocalLoss_(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


class cross_entropy(torch.nn.Module):
    def __init__(self):
        super(cross_entropy, self).__init__()

    def forward(self, input, target, size_average=True):
        if size_average:
            return torch.mean(torch.sum(-target * F.log_softmax(input, dim=1), dim=1))
        else:
            return torch.sum(torch.sum(-target * F.log_softmax(input, dim=1), dim=1))


def BCEWithLogitsLoss(config):
    return nn.BCEWithLogitsLoss()


def FocalLoss(config):
    return FocalLoss_(gamma=config['model']['loss']['gamma'])


def MAE(config):
    return nn.L1Loss()


def MSE(config):
    return nn.MSELoss()


class Lq_(torch.nn.Module):
    def __init__(self, q=0.3):
        self.q = q
        super(Lq_, self).__init__()

    def forward(self, input, target):
        return torch.mean((1 - ((input * target).sum(1, keepdim=True))**self.q) / self.q)


def Lq(config):
    return Lq_()
