import json
import time
import pathlib
import argparse
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torchvision import transforms
from torch.utils.data import DataLoader

from src.utils import get_module_logger, seed_everything, save_json, calculate_per_class_lwlrap, _one_sample_positive_class_precisions
from src.edit_data import get_silent_wav_list
from src.data_loader import ToTensor, get_meta_data, FAT_TrainSet_logmel, FAT_TestSet_logmel, FAT_ValSet_logmel
from src.data_transform import wav_to_logmel

from src.networks.simple_2d_cnn import simple_2d_cnn_logmel
from src.loss_func import BCEWithLogitsLoss, FocalLoss, MAE, MSE, Lq, CrossEntropyOneHot
from src.optimizers import opt_Adam, opt_SGD, opt_AdaBound
from src.schedulers import sche_CosineAnnealingLR
from src.train import train_on_fold, predict_model

MODEL_map = {
    'simple_2d_cnn_logmel': simple_2d_cnn_logmel
}

LOSS_map = {
    'BCEWithLogitsLoss': BCEWithLogitsLoss,
    'FocalLoss': FocalLoss,
    'CrossEntropyOneHot': CrossEntropyOneHot,
    'MAE': MAE,
    'MSE': MSE,
    'Lq': Lq
}

OPTIMIZER_map = {
    'Adam': opt_Adam,
    'SGD': opt_SGD,
    'AdaBound': opt_AdaBound
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
    config.update({
        'args': {
            'config': args.config,
            'debug_mode': args.debug,
            'force': args.force
        }
    })

    # make model-output-dir
    output_dir = config['dataset']['output_directory']
    model_no = args.config.split('/')[-1].split('.')[0]
    model_output_dir = pathlib.Path(output_dir + model_no + '/')
    if not model_output_dir.exists():
        model_output_dir.mkdir()
    logger.info(f'model_output_dir: {str(model_output_dir)}')
    logger.debug(f'model_output_dir exists: {model_output_dir.exists()}')
    config.update({
        'model_output_dir': str(model_output_dir)
    })

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
            wav_to_logmel(train_wavelist, config, fe_dir)
            logger.debug(f'test data processing. {len(test_wavelist)}')
            wav_to_logmel(test_wavelist, config, fe_dir)

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
        if config['pre-processing']['data-selection']['name'] == 'ALL':
            use_index = train.index

        elif config['pre-processing']['data-selection']['name'] == 'ONLY_CURATED':
            silent_wav_list = get_silent_wav_list()
            idxes_curated = train.query('noisy_flg == 0 and fname not in @silent_wav_list').index.values
            idxes_noisy = train.query('noisy_flg == 1 and fname not in @silent_wav_list').index.values
            use_index = idxes_curated

    #######################################################################################################
    train_noisy = train.iloc[idxes_noisy].reset_index(drop=True)
    y_train_noisy = y_train[idxes_noisy]
    #######################################################################################################

    train = train.iloc[use_index].reset_index(drop=True)
    y_train = y_train[use_index]

    #######################################################################################################
    # scaling of target
    #y_train = y_train.astype(float)
    #for i in range(len(y_train)):
    #    y_train[i] = y_train[i] / y_train[i].sum()
    #######################################################################################################

    #######################################################################################################
    # get true noisy
    preds_all = np.zeros((len(train_noisy), len(labels))).astype(np.float32)
    for i_fold_tmp in range(config['cv']['n_splits']):
        # define train-loader and valid-loader
        if feature_name == 'logmel':
            val_transform = transforms.Compose([
                ToTensor()
            ])
            valSet = FAT_ValSet_logmel(
                config=config, dataframe=train_noisy, labels=y_train_noisy,
                transform=val_transform, fnames=train_noisy['fname']
            )
        val_loader = DataLoader(
            valSet, batch_size=config['model']['params']['batch_size'],
            shuffle=False,
            num_workers=config['model']['params']['num_workers']
        )
        # load model
        model = MODEL_map[config['model']['name']]()
        model.load_state_dict(torch.load(f'./data/output/model_32/weight_best_fold{i_fold_tmp+1}.pt'))
        model.cuda()
        model.eval()

        preds_list_tmp = []
        fname_list = []
        with torch.no_grad():
            for i, (x_batch, y_batch, fnames) in enumerate(val_loader):
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda(non_blocking=True)
                output = model(x_batch)
                preds_list_tmp.append(torch.sigmoid(output).cpu().numpy())
                fname_list.extend(fnames)

        val_preds = pd.DataFrame(
            data=np.concatenate(preds_list_tmp),
            index=fname_list,
            columns=map(str, range(len(labels)))
        )
        val_preds = val_preds.groupby(level=0).mean()   # group by fname
        preds_all = preds_all + val_preds.values / (config['cv']['n_splits'])

    # calc lwlrap per samples
    train_noisy_result = pd.DataFrame()
    train_noisy_fname_list = train_noisy['fname'].tolist()
    for i in range(len(train_noisy_fname_list)):
        fname = train_noisy_fname_list[i]
        pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(preds_all[i], y_train_noisy[i])
        for i_class in range(len(pos_class_indices)):
            class_name = labels[pos_class_indices[i_class]]
            precision = precision_at_hits[i_class]
            train_noisy_result = train_noisy_result.append([[fname, class_name, precision]])
    train_noisy_result.columns = ['fname', 'class_name', 'precision_at_hits']
    train_noisy_result = pd.merge(train_noisy_result, train_noisy, on="fname", how="inner")

    precision_per_fname = train_noisy_result.groupby("fname")["precision_at_hits"].mean().reset_index()
    threshold = 1.0
    use_index_noisy = precision_per_fname.query("precision_at_hits >= @threshold").index

    train_noisy = train_noisy.iloc[use_index_noisy]
    y_train_noisy = y_train_noisy[use_index_noisy]
    #######################################################################################################

    logger.info(f'n_use_train_data: {len(train)}')

    # set fold
    if config['cv']['method'] == 'MultilabelStratifiedKFold':
        skf = MultilabelStratifiedKFold(
            n_splits=config['cv']['n_splits'],
            shuffle=config['cv']['shuffle'],
            random_state=config['cv']['random_state']
        )
    elif config['cv']['method'] == 'KFold':
        skf = KFold(
            n_splits=config['cv']['n_splits'],
            shuffle=config['cv']['shuffle'],
            random_state=config['cv']['random_state']
        )

    # train
    for i_fold, (trn_idx, val_idx) in enumerate(skf.split(train, y_train)):
        end = time.time()

        # split dataset
        trn_set = train.iloc[trn_idx].reset_index(drop=True)
        y_trn = y_train[trn_idx]
        val_set = train.iloc[val_idx].reset_index(drop=True)
        y_val = y_train[val_idx]

        #######################################################################################################
        # get psuedo label
        """
        preds_all = np.zeros((len(train_noisy), len(labels))).astype(np.float32)
        for i_fold_tmp in range(config['cv']['n_splits']):
            if i_fold_tmp == i_fold:
                continue

            # define train-loader and valid-loader
            if feature_name == 'logmel':
                val_transform = transforms.Compose([
                    ToTensor()
                ])
                valSet = FAT_ValSet_logmel(
                    config=config, dataframe=train_noisy, labels=y_train_noisy,
                    transform=val_transform, fnames=train_noisy['fname']
                )
            val_loader = DataLoader(
                valSet, batch_size=config['model']['params']['batch_size'],
                shuffle=False,
                num_workers=config['model']['params']['num_workers']
            )
            # load model
            model = MODEL_map[config['model']['name']]()
            model.load_state_dict(torch.load(f'./data/output/model_29/weight_best_fold{i_fold_tmp+1}.pt'))
            model.cuda()
            model.eval()

            preds_list_tmp = []
            fname_list = []
            with torch.no_grad():
                for i, (x_batch, y_batch, fnames) in enumerate(val_loader):
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda(non_blocking=True)
                    output = model(x_batch)
                    preds_list_tmp.append(torch.sigmoid(output).cpu().numpy())
                    fname_list.extend(fnames)

            val_preds = pd.DataFrame(
                data=np.concatenate(preds_list_tmp),
                index=fname_list,
                columns=map(str, range(len(labels)))
            )
            val_preds = val_preds.groupby(level=0).mean()   # group by fname
            preds_all = preds_all + val_preds.values / (config['cv']['n_splits'] - 1)
        """
        #######################################################################################################

        #######################################################################################################
        trn_set = pd.concat([trn_set, train_noisy], axis=0, ignore_index=True, sort=False)
        # y_trn = np.concatenate((y_trn, y_train_noisy))
        y_trn = np.concatenate((y_trn, y_train_noisy))
        #######################################################################################################

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
                transform=train_transform
            )
            valSet = FAT_TrainSet_logmel(
                config=config, dataframe=val_set, labels=y_val,
                transform=val_transform
            )

        trn_loader = DataLoader(
            trnSet, batch_size=config['model']['params']['batch_size'],
            shuffle=config['model']['params']['shuffle'],
            num_workers=config['model']['params']['num_workers']
        )
        val_loader = DataLoader(
            valSet, batch_size=config['model']['predict']['test_batch_size'],
            shuffle=False,
            num_workers=config['model']['params']['num_workers']
        )

        # load model
        model = MODEL_map[config['model']['name']]()
        #######################################################################################################
        #model.load_state_dict(torch.load(f'./data/output/model_34/weight_best_fold{i_fold+1}.pt'))
        #######################################################################################################
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
    # === Check Train Result
    # =========================================
    # check total score
    preds_list = []
    target_list = []
    val_fname_list = []
    for i_fold, (trn_idx, val_idx) in enumerate(skf.split(train, y_train)):
        val_set = train.iloc[val_idx].reset_index(drop=True)
        y_val = y_train[val_idx]
        val_fname_list.extend(val_set['fname'].tolist())

        # define train-loader and valid-loader
        if feature_name == 'logmel':
            val_transform = transforms.Compose([
                ToTensor()
            ])
            valSet = FAT_TrainSet_logmel(
                config=config, dataframe=val_set, labels=y_val,
                transform=val_transform
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
        model.eval()

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(val_loader):
                if config['model']['params']['cuda']:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda(non_blocking=True)
                output = model(x_batch)
                preds_list.append(torch.sigmoid(output).cpu().numpy())
                target_list.append(y_batch.cpu().numpy())

    # summary
    all_preds = np.concatenate(preds_list)
    all_target = np.concatenate(target_list)
    score, weight = calculate_per_class_lwlrap(all_target, all_preds)
    lwlrap = (score * weight).sum()
    logger.info(f'total lwlrap: {lwlrap}')
    config.update({'total': {
        'best_lwlrap': lwlrap
    }})

    # calc lwlrap per samples
    val_result = pd.DataFrame()
    for i in range(len(val_fname_list)):
        fname = val_fname_list[i]
        pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(all_preds[i], all_target[i])
        for i_class in range(len(pos_class_indices)):
            class_name = labels[pos_class_indices[i_class]]
            precision = precision_at_hits[i_class]
            val_result = val_result.append([[fname, class_name, precision]])
    val_result.columns = ['fname', 'class_name', 'precision_at_hits']
    val_result = pd.merge(val_result, train, on="fname", how="inner")
    val_result.to_csv(model_output_dir / 'val_result.csv', index=False)

    # =========================================
    # === Check Train Result (tmp)
    # =========================================
    """
    # check total score
    preds_list = []
    target_list = []
    val_fname_list = []
    for i_fold, (trn_idx, val_idx) in enumerate(skf.split(train, y_train)):
        val_set = train.iloc[val_idx].reset_index(drop=True)
        y_val = y_train[val_idx]
        val_fname_list.extend(val_set['fname'].tolist())

        # define train-loader and valid-loader
        if feature_name == 'logmel':
            val_transform = transforms.Compose([
                ToTensor()
            ])
            valSet = FAT_ValSet_logmel(
                config=config, dataframe=val_set, labels=y_val,
                transform=val_transform, fnames=val_set['fname']
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
        model.eval()

        preds_list_tmp = []
        fname_list = []
        with torch.no_grad():
            for i, (x_batch, y_batch, fnames) in enumerate(val_loader):
                if config['model']['params']['cuda']:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda(non_blocking=True)
                output = model(x_batch)
                preds_list_tmp.append(torch.sigmoid(output).cpu().numpy())
                fname_list.extend(fnames)

        val_preds = pd.DataFrame(
            data=np.concatenate(preds_list_tmp),
            index=fname_list,
            columns=map(str, range(len(labels)))
        )
        from scipy.stats import mstats
        val_preds = val_preds.groupby(level=0).max()   # group by fname
        #val_preds = val_preds.groupby(level=0).mean()   # group by fname
        #val_preds = val_preds.groupby(level=0).agg(mstats.gmean)   # group by fname
        preds_list.append(val_preds.values)
        target_list.append(y_val)

    # summary
    all_preds = np.concatenate(preds_list)
    all_target = np.concatenate(target_list)
    score, weight = calculate_per_class_lwlrap(all_target, all_preds)
    lwlrap = (score * weight).sum()
    logger.info(f'total lwlrap: {lwlrap}')
    config.update({'total': {
        'best_lwlrap': lwlrap
    }})

    # calc lwlrap per samples
    val_result = pd.DataFrame()
    for i in range(len(val_fname_list)):
        fname = val_fname_list[i]
        pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(all_preds[i], all_target[i])
        for i_class in range(len(pos_class_indices)):
            class_name = labels[pos_class_indices[i_class]]
            precision = precision_at_hits[i_class]
            val_result = val_result.append([[fname, class_name, precision]])
    val_result.columns = ['fname', 'class_name', 'precision_at_hits']
    val_result = pd.merge(val_result, train, on="fname", how="inner")
    val_result.to_csv(model_output_dir / 'val_result.csv', index=False)
    """

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
