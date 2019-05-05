import json
import time
import pathlib
import argparse
import random
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torch.utils.data import DataLoader

from src.utils import get_module_logger, seed_everything, save_json, calculate_per_class_lwlrap
from src.data_loader import get_meta_data, ToTensor, FAT_TrainSet_logmel, FAT_TestSet_logmel
from src.data_transform import wav_to_logmel
from src.networks.simple_2d_cnn import simple_2d_cnn_logmel
from src.loss_func import BCEWithLogitsLoss, FocalLoss
from src.optimizers import opt_Adam, opt_SGD
from src.schedulers import sche_CosineAnnealingLR
from src.train import train_on_fold, predict_model

MODEL_map = {
    'simple_2d_cnn_logmel': simple_2d_cnn_logmel
}

LOSS_map = {
    'BCEWithLogitsLoss': BCEWithLogitsLoss,
    'FocalLoss': FocalLoss
}

OPTIMIZER_map = {
    'Adam': opt_Adam,
    'SGD': opt_SGD
}

SCHEDULER_map = {
    'CosineAnnealingLR': sche_CosineAnnealingLR
}


def main():
    # =========================================
    # === Settings
    # =========================================

    # get logger
    logger = get_module_logger(__name__)

    # get argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/model_0.json')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    logger.info(f'config: {args.config}')
    logger.info(f'debug: {args.debug}')
    logger.info(f'force: {args.force}')

    # get config
    config = json.load(open(args.config))
    args_log = {'args': {
        'config': args.config,
        'debug_mode': args.debug,
        'force': args.force
    }}
    config.update(args_log)

    # make model-output-dir
    output_dir = config['dataset']['output_directory']
    model_no = args.config.split('/')[-1].split('.')[0]
    model_output_dir = pathlib.Path(output_dir + model_no + '/')
    if not model_output_dir.exists():
        model_output_dir.mkdir()
    logger.info(f'model_output_dir: {str(model_output_dir)}')
    logger.debug(f'model_output_dir exists: {model_output_dir.exists()}')
    args_log = {'model_output_dir': str(model_output_dir)}
    config.update(args_log)

    # fix seed
    seed_everything(71)

    # =========================================
    # === Data Processing
    # =========================================

    # get meta-data
    train, test, y_train, labels = get_meta_data(config)
    logger.info(f'n_classes: {len(labels)}')

    # get features
    feature_name = config['features']['name']
    logger.info(f'feature_name: {feature_name}')

    input_dir = config['dataset']['input_directory']
    train_wavelist = [
        input_dir + 'train_curated/' + row['fname'] if row['noisy_flg'] == 0
        else input_dir + 'train_noisy/' + row['fname'] for ind, row in train.iterrows()
    ]
    test_wavelist = [
        input_dir + 'test/' + row['fname'] for ind, row in test.iterrows()
    ]

    # make features
    if feature_name == 'raw_wave':
        pass

    elif feature_name == 'logmel':
        fe_params = config['features']['params']
        duration = fe_params['duration']
        n_mels = fe_params['n_mels']
        n_fft = fe_params['factor__n_fft'] * n_mels
        fe_dir = pathlib.Path(
            config['dataset']['intermediate_directory'] +
            f'logmel+delta_nmels{n_mels}_duration{duration}_nfft{n_fft}/'
        )
        if not fe_dir.exists():
            fe_dir.mkdir()
        logger.info(f'fe_dir: {str(fe_dir)}')
        logger.debug(f'fe_dir exists: {fe_dir.exists()}')
        args_log = {'fe_dir': str(fe_dir)}
        config.update(args_log)

        if args.force:
            logger.info('make features.')
            logger.debug(f'train data processing. {len(train_wavelist)}')
            wav_to_logmel(train_wavelist, fe_params, fe_dir)
            logger.debug(f'test data processing. {len(test_wavelist)}')
            wav_to_logmel(test_wavelist, fe_params, fe_dir)

    elif feature_name == 'mfcc':
        pass

    # =========================================
    # === Train Model
    # =========================================

    # choose training data
    if args.debug:
        # use only few data
        use_index = train.index[:500]
    else:
        if config['pre-processing']['data-selection']['name'] == 'NOISY_BEST50S':
            # get single-label of noisy data
            # https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
            noisy_single_df = train.query('noisy_flg == 1 & n_labels == 1')
            noisy_labels = noisy_single_df.labels.unique().tolist()
            idxes_best50s = np.array([random.choices(noisy_single_df[(noisy_single_df.labels == l)].index, k=50) for l in labels]).ravel()
            idxes_curated = train.query('noisy_flg == 0').index.values
            use_index = np.concatenate((idxes_curated, idxes_best50s))
        else:
            # use all data
            use_index = train.index

    train = train.iloc[use_index].reset_index(drop=True)
    y_train = y_train[use_index]
    logger.info(f'n_use_train_data: {len(use_index)}')

    # set fold
    if config['cv']['method'] == 'StratifiedKFold':
        skf = StratifiedKFold(
            n_splits=config['cv']['n_splits'],
            shuffle=config['cv']['shuffle'],
            random_state=config['cv']['random_state']
        )

    # create labels for splits
    label_idx = {label: i for i, label in enumerate(labels)}
    train['first_label'] = train['labels'].apply(lambda x: x.split(',')[0])
    train["label_idx"] = train['first_label'].apply(lambda x: label_idx[x])

    # train
    for i_fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['label_idx'])):
        end = time.time()

        # split dataset
        trn_set = train.iloc[trn_idx].reset_index(drop=True)
        y_trn = y_train[trn_idx]
        val_set = train.iloc[val_idx].reset_index(drop=True)
        y_val = y_train[val_idx]
        logger.info(f'Fold {i_fold+1}, train samples: {len(trn_set)}, val samples: {len(val_set)}')

        # define train-loader and valid-loader
        if feature_name == 'logmel':
            train_transform = transforms.Compose([
                ToTensor()
            ])
            val_transform = transforms.Compose([
                ToTensor()
            ])
            trnSet = FAT_TrainSet_logmel(
                config=config, dataframe=trn_set, labels=y_trn,
                specAug=config['model']['specAug']['enabled'], transform=train_transform
            )
            valSet = FAT_TrainSet_logmel(
                config=config, dataframe=val_set, labels=y_val,
                specAug=False, transform=val_transform
            )

        trn_loader = DataLoader(
            trnSet, batch_size=config['model']['params']['batch_size'],
            shuffle=config['model']['params']['shuffle'],
            num_workers=config['model']['params']['num_workers']
        )
        val_loader = DataLoader(
            valSet, batch_size=config['model']['params']['batch_size'],
            shuffle=False,
            num_workers=config['model']['params']['num_workers']
        )

        # load model
        model = MODEL_map[config['model']['name']]()
        if config['model']['params']['cuda']:
            model.cuda()

        # setting train parameters
        criterion = LOSS_map[config['model']['loss']['name']](config).cuda()
        optimizer = OPTIMIZER_map[config['model']['optimizer']['name']](model.parameters(), config)
        scheduler = SCHEDULER_map[config['model']['scheduler']['name']](optimizer, config)
        torch.backends.cudnn.benchmark = True

        # train model
        train_on_fold(
            model, trn_loader, val_loader,
            criterion, optimizer, scheduler, config, i_fold, logger
        )

        time_on_fold = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - end))
        logger.info(f'--------------Time on fold {i_fold+1}: {time_on_fold}--------------\n')

    # =========================================
    # === Tmp
    # =========================================

    # check total score
    preds_list = []
    target_list = []
    for i_fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['label_idx'])):
        val_set = train.iloc[val_idx].reset_index(drop=True)
        y_val = y_train[val_idx]

        # define train-loader and valid-loader
        if feature_name == 'logmel':
            val_transform = transforms.Compose([
                ToTensor()
            ])
            valSet = FAT_TrainSet_logmel(
                config=config, dataframe=val_set, labels=y_val,
                specAug=False, transform=val_transform
            )
        val_loader = DataLoader(
            valSet, batch_size=config['model']['params']['batch_size'],
            shuffle=False,
            num_workers=config['model']['params']['num_workers']
        )
        # load model
        model = MODEL_map[config['model']['name']]()
        if config['model']['params']['cuda']:
            model.load_state_dict(torch.load(model_output_dir / f'weight_best_fold{i_fold+1}.pt'))
            model.cuda()
        else:
            model.load_state_dict(torch.load(model_output_dir / f'weight_best_fold{i_fold+1}.pt', map_location='cpu'))
        model.eval()

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(val_loader):
                if config['model']['params']['cuda']:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda(non_blocking=True)
                output = model(x_batch)
                preds_list.append(torch.sigmoid(output).cpu().numpy())
                # preds_list.append(output.cpu().numpy())
                target_list.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(preds_list)
    all_target = np.concatenate(target_list)
    score, weight = calculate_per_class_lwlrap(all_target, all_preds)
    lwlrap = (score * weight).sum()
    logger.info(f'total lwlrap: {lwlrap}')
    config.update({'total': {
        'best_lwlrap': lwlrap
    }})

    # =========================================
    # === Predict
    # =========================================
    n_splits = config['cv']['n_splits']
    test_preds = np.zeros_like(test[labels].values).astype(np.float32)

    test_transform = transforms.Compose([
        ToTensor()
    ])
    testSet = FAT_TestSet_logmel(
        config=config, fnames=test['fname'], transform=test_transform,
        tta=config['model']['predict']['tta']
    )
    test_loader = DataLoader(
        testSet, batch_size=config['model']['predict']['test_batch_size'],
        shuffle=False,
        num_workers=config['model']['predict']['num_workers']
    )

    for i_fold in range(n_splits):
        # load trained model
        model_save_name = model_output_dir / f'weight_best_fold{i_fold+1}.pt'
        model = MODEL_map[config['model']['name']]()
        model.load_state_dict(torch.load(model_save_name))
        if config['model']['params']['cuda']:
            model.cuda()

        test_preds_one_fold = predict_model(model, test_loader, len(labels))
        test_preds_one_fold = test_preds_one_fold.loc[test['fname']].values
        test_preds += test_preds_one_fold / n_splits

    test[labels] = test_preds
    test.to_csv(model_output_dir / 'submission.csv', index=False)

    # =========================================
    # === Save
    # =========================================
    save_path = model_output_dir / 'output.json'
    save_json(config, save_path, logger)


if __name__ == '__main__':
    main()
