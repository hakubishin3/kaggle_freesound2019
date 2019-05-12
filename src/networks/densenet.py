from torchvision.models.densenet import densenet121


def densenet121_logmel():
    model = densenet121(
        pretrained=False,
        drop_rate=0.2,
        num_classes=80
    )
    return model
