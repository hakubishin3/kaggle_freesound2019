import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
class FocalLoss_(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()


def BCEWithLogitsLoss(config):
    # return nn.BCELoss()
    return nn.BCEWithLogitsLoss()


def FocalLoss(config):
    return FocalLoss_(gamma=config['model']['loss']['gamma'])
