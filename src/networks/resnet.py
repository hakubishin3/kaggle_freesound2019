import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101


def resnet101_logmel():
    model = resnet101(
        pretrained=False,
        num_classes=80
    )
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model.avgpool = nn.AvgPool2d((2, 5), stride=(2, 5))
    # model.fc = nn.Linear(512 * 8, 80)

    return model
