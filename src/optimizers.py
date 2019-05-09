from torch.optim import Adam, SGD


def opt_Adam(model_params, config):
    optimizer = Adam(
        params=model_params,
        lr=config['model']['optimizer']['lr'],
        amsgrad=config['model']['optimizer']['amsgrad']
    )
    return optimizer


def opt_SGD(model_params, config):
    optimizer = SGD(
        params=model_params,
        lr=config['model']['optimizer']['lr'],
        weight_decay=config['model']['optimizer']['weight_decay'],
        momentum=config['model']['optimizer']['momentum'],
        nesterov=config['model']['optimizer']['nesterov']
    )
    return optimizer