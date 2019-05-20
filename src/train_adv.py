import time
import torch
import random
import pathlib
import numpy as np
import pandas as pd
from .utils import EarlyStopping
from sklearn.metrics import roc_auc_score


def train_on_fold(model, trn_loader, val_loader,
                  criterion, optimizer, scheduler, config, i_fold, logger):
    best_auc = 0
    best_loss = 0
    best_auc_train = 0
    best_loss_train = 0
    train_result = {}

    # initialize the early_stopping object
    patience = config['model']['params']['early_stopping_patience']
    filename = pathlib.Path(config['model_output_dir']) / f'weight_best_fold{i_fold+1}.pt'
    early_stopping = EarlyStopping(filename=filename, patience=patience, verbose=True)
    logger.info(f'model_name: {filename}')

    for i_epoch in range(config['model']['params']['n_epochs']):
        end = time.time()

        if config['model']['scheduler']['enabled']:
            scheduler.step()

        # train for one epoch
        train_losses, train_auc = train_one_epoch(model, trn_loader, criterion, optimizer, config)

        # evaluate on validation set
        val_losses, val_auc = val_on_fold(model, val_loader, criterion, config)

        elapsed_time = time.time() - end
        if (i_epoch + 1) % config['model']['params']['print_epoch'] == 0:
            new_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Fold {i_fold+1} Epoch {i_epoch+1}: Time {elapsed_time:.0f}s, lr {new_lr:.4g}, train_losses:{train_losses:.4f}, train_auc:{train_auc:.4f}, val_losses:{val_losses:.4f}, val_auc:{val_auc:.4f}')

        if val_auc > best_auc:
            best_epoch = i_epoch + 1
            best_auc = val_auc
            best_loss = val_losses
            best_auc_train = train_auc
            best_loss_train = train_losses

        # early_stopping needs the validation loss to check if it has decresed, and if it has, it will make a checkpoint of the current model
        early_stopping(val_auc, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # summary
    best_auc = early_stopping.best_score
    config.update({
        f'train_result_fold{i_fold+1}': {
            'best_epoch': best_epoch,
            'best_auc': best_auc,
            'best_loss': best_loss,
            'best_auc_train': best_auc_train,
            'best_loss_train': best_loss_train
        }
    })


def train_one_epoch(model, trn_loader, criterion, optimizer, config):
    losses = 0.
    preds_list = []
    target_list = []

    # switch to train mode
    model.train()

    for i, (x_batch, y_batch) in enumerate(trn_loader):
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        output = model(x_batch)
        loss = criterion(output, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item() / len(trn_loader)

        preds_list.append(torch.sigmoid(output).cpu().detach().numpy())
        target_list.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(preds_list)
    all_target = np.concatenate(target_list)
    score = roc_auc_score(all_target, all_preds)

    return losses, score


def val_on_fold(model, val_loader, criterion, config):
    losses = 0.
    preds_list = []
    target_list = []

    # swith to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(val_loader):
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda(non_blocking=True)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            losses += loss.item() / len(val_loader)

            preds_list.append(torch.sigmoid(output).cpu().numpy())
            target_list.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(preds_list)
    all_target = np.concatenate(target_list)
    score = roc_auc_score(all_target, all_preds)

    return losses, score
