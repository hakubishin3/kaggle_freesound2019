from torchvision import models
import torch.nn as nn


def resnet50_logmel(pretrained=False, **kwargs):
    model = models.resnet50(
        pretrained=pretrained,
        num_classes=80
    )
    return model
