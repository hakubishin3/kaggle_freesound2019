import time
import torch
import pathlib
import numpy as np
import pandas as pd
from .utils import get_module_logger, calculate_per_class_lwlrap, seed_everything


def train_on_fold(model, trn_loader, val_loader,
                  criterion, optimizer, scheduler, config, i_fold, logger):
    # get logger
    # logger = get_module_logger(__name__)

    best_lwlrap = 0
    train_result = {}

    for i_epoch in range(config['model']['params']['n_epochs']):
        end = time.time()
        scheduler.step()

        # train for one epoch
        train_losses, train_lwlrap = train_one_epoch(model, trn_loader, criterion, optimizer, config)

        # evaluate on validation set
        val_losses, val_lwlrap = val_on_fold(model, val_loader, criterion, config)

        elapsed_time = time.time() - end
        if (i_epoch + 1) % config['model']['params']['print_epoch'] == 0:
            new_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Fold {i_fold+1} Epoch {i_epoch+1}: Time {elapsed_time:.0f}s, lr {new_lr:.4g}, train_losses:{train_losses:.4f}, train_lwlrap:{train_lwlrap:.4f}, val_losses:{val_losses:.4f}, val_lwlrap:{val_lwlrap:.4f}')

        if val_lwlrap > best_lwlrap:
            logger.info(f'updated: {best_lwlrap} -> {val_lwlrap}')
            best_epoch = i_epoch + 1
            best_lwlrap = val_lwlrap
            torch.save(model.state_dict(), pathlib.Path(config['model_output_dir']) / f'weight_best_fold{i_fold+1}.pt')

    config.update({
        f'train_result_fold{i_fold+1}': {
            'best_epoch': best_epoch,
            'best_lwlrap': best_lwlrap
        }
    })


def train_one_epoch(model, trn_loader, criterion, optimizer, config):
    losses = 0.
    preds_list = []
    target_list = []

    # switch to train mode
    model.train()

    for i, (x_batch, y_batch) in enumerate(trn_loader):
        if config['model']['mixup']['enabled']:
            x_batch, y_batch = mixup(x_batch, y_batch, alpha=config['model']['mixup']['alpha'])

        if config['model']['params']['cuda']:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        output = model(x_batch)
        loss = criterion(output, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item() / len(trn_loader)

        preds_list.append(torch.sigmoid(output).cpu().detach().numpy())
        # preds_list.append(output.cpu().detach().numpy())
        target_list.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(preds_list)
    all_target = np.concatenate(target_list)
    score, weight = calculate_per_class_lwlrap(all_target, all_preds)
    lwlrap = (score * weight).sum()

    return losses, lwlrap


def val_on_fold(model, val_loader, criterion, config):
    losses = 0.
    preds_list = []
    target_list = []

    # swith to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(val_loader):
            if config['model']['params']['cuda']:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda(non_blocking=True)

            output = model(x_batch)
            loss = criterion(output, y_batch)
            losses += loss.item() / len(val_loader)

            preds_list.append(torch.sigmoid(output).cpu().numpy())
            # preds_list.append(output.cpu().numpy())
            target_list.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(preds_list)
    all_target = np.concatenate(target_list)
    score, weight = calculate_per_class_lwlrap(all_target, all_preds)
    lwlrap = (score * weight).sum()

    return losses, lwlrap


def predict_model(model, test_loader, n_classes):
    # switch to evaluate mode
    model.eval()

    preds_list = []
    fname_list = []
    with torch.no_grad():
        for x_batch, fnames in test_loader:
            x_batch = x_batch.cuda()
            output = model(x_batch)
            preds_list.append(torch.sigmoid(output).cpu().numpy())
            # preds_list.append(output.cpu().numpy())
            fname_list.extend(fnames)

    test_preds = pd.DataFrame(
        data=np.concatenate(preds_list),
        index=fname_list,
        columns=map(str, range(n_classes))
    )
    test_preds = test_preds.groupby(level=0).mean()   # group by fname

    return test_preds


def mixup(data, one_hot_labels, alpha=1):
    batch_size = data.size()[0]
    weights = np.random.beta(alpha, alpha, batch_size)
    weights = torch.from_numpy(weights).type(torch.FloatTensor)

    index = np.random.permutation(batch_size)
    x1, x2 = data, data[index]

    x = torch.zeros_like(x1)
    for i in range(batch_size):
        for c in range(x.size()[1]):
            x[i][c] = x1[i][c] * weights[i] + x2[i][c] * (1 - weights[i])

    y1 = one_hot_labels
    y2 = one_hot_labels[index]
    y = torch.zeros_like(y1)

    for i in range(batch_size):
        y[i] = y1[i] * weights[i] + y2[i] * (1 - weights[i])

    return x, y
