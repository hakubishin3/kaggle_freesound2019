from torch.optim.lr_scheduler import CosineAnnealingLR


def sche_CosineAnnealingLR(optimizer, config):
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['model']['scheduler']['T_max'],
        eta_min=config['model']['scheduler']['eta_min']
    )
    return scheduler
