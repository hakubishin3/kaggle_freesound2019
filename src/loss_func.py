import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss_(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduction='elementwise_mean'):
        super(FocalLoss_, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss
        else:
            return torch.mean(F_loss)


class cross_entropy(torch.nn.Module):
    """
    Cross entropy  that accepts soft targets (like [0, 0.1, 0.1, 0.8, 0]).
    """
    def __init__(self):
        super(cross_entropy, self).__init__()

    def forward(self, input, target, size_average=True):
        if size_average:
            return torch.mean(torch.sum(-target * F.log_softmax(input, dim=1), dim=1))
        else:
            return torch.sum(torch.sum(-target * F.log_softmax(input, dim=1), dim=1))


def CrossEntropyOneHot(config):
    return cross_entropy()


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
