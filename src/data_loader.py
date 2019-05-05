import os
import numpy as np
import pandas as pd
import random
import torch
import pathlib
import torch
from torch.utils.data import Dataset
from .utils import load_data
from .data_transform import standarize
from PIL import Image


def get_meta_data(config):
    data_dir = config['dataset']['input_directory']
    config_dataset_files_meta = config['dataset']['files']['meta']
    train_curated = pd.read_csv(data_dir + config_dataset_files_meta['train_curated'])
    train_noisy = pd.read_csv(data_dir + config_dataset_files_meta['train_noisy'])
    test = pd.read_csv(data_dir + config_dataset_files_meta['test'])

    train = pd.concat([train_curated, train_noisy], axis=0, sort=False, ignore_index=True)
    train['noisy_flg'] = 0
    train.loc[len(train_curated):len(train_curated) + len(train_noisy), 'noisy_flg'] = 1
    train['n_labels'] = train['labels'].apply(lambda x: len(x.split(',')))

    labels = test.columns[1:].tolist()
    n_classes = len(labels)
    y_train = np.zeros((len(train), n_classes)).astype(int)
    for i, row in enumerate(train['labels'].str.split(',')):
        for label in row:
            idx = labels.index(label)
            y_train[i, idx] = 1

    return train, test, y_train, labels


class ToTensor(object):
    """
    convert ndarrays in sample to Tensors.
    return:
        feat(torch.FloatTensor)
        label(torch.LongTensor of size batch_size x 1)
    """
    def __call__(self, data):
        data = torch.from_numpy(data).type(torch.FloatTensor)
        return data


def freq_mask(data, F=30, num_masks=1, replace_with_zero=False):
    spec = data.copy()   # data.shape = (3, n_mel, xx)
    num_mel_channels = spec.shape[1]

    mask_value_list = []
    for i_channel in range(spec.shape[0]):
        mask_value_list.append(spec[i_channel].mean())

    for i in range(num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        if f_zero == f_zero + f:
            # avoids randrange error if values are equal and range is empty
            return spec

        mask_end = random.randrange(f_zero, f_zero + f)
        if replace_with_zero:
            for i_channel in range(spec.shape[0]):
                spec[i_channel, f_zero:mask_end, :] = 0
        else:
            for i_channel in range(spec.shape[0]):
                # mask_value = np.random.rand()
                mask_value = mask_value_list[i_channel]
                spec[i_channel, f_zero:mask_end, :] = mask_value

    return spec


def time_mask(data, T=30, num_masks=1, replace_with_zero=False):
    spec = data.copy()   # data.shape = (3, n_mel, xx)
    len_spectro = spec.shape[2]

    mask_value_list = []
    for i_channel in range(spec.shape[0]):
        mask_value_list.append(spec[i_channel].mean())

    for i in range(num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        if t_zero == t_zero + t:
            # avoids randrange error if values are equal and range is empty
            return spec

        mask_end = random.randrange(t_zero, t_zero + t)
        if replace_with_zero:
            for i_channel in range(spec.shape[0]):
                spec[i_channel, :, t_zero:mask_end] = 0
        else:
            for i_channel in range(spec.shape[0]):
                # mask_value = np.random.rand()
                mask_value = mask_value_list[i_channel]
                spec[i_channel, :, t_zero:mask_end] = mask_value

    return spec


class FAT_TrainSet_logmel(Dataset):
    def __init__(self, config, dataframe, labels, specAug, transform=None):
        self.config = config
        self.dataframe = dataframe
        self.labels = labels
        self.transform = transform
        self.specAug = specAug

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        fe_dir = pathlib.Path(self.config['fe_dir'])
        filename = os.path.splitext(self.dataframe["fname"][idx])[0] + '.pkl'
        file_path = fe_dir / filename

        # Read and Resample the audio
        data = self._random_selection(file_path)
        label = torch.from_numpy(self.labels[idx]).float()

        # specAugment
        # ※ normalizeの後に実行すること
        if self.specAug:
            F = self.config['model']['specAug']['F']
            F_num_masks = self.config['model']['specAug']['F_num_masks']
            T = self.config['model']['specAug']['T']
            T_num_masks = self.config['model']['specAug']['T_num_masks']
            replace_with_zero = self.config['model']['specAug']['replace_with_zero']
            data = time_mask(
                freq_mask(data, F=F, num_masks=F_num_masks, replace_with_zero=replace_with_zero),
                T=T, num_masks=T_num_masks, replace_with_zero=replace_with_zero
            )

        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def _random_selection(self, file_path):
        input_length = self.config['model']['params']['input_length']

        # Read the logmel pkl
        logmel = load_data(file_path)

        # standarize([0, 1])
        for i_channel in range(logmel.shape[0]):
            logmel[i_channel, :, :] = standarize(logmel[i_channel, :, :])

        # Random offset / Padding
        if logmel.shape[2] > input_length:   # logmel.shape = (3, n_mel, xx)
            max_offset = logmel.shape[2] - input_length
            offset = np.random.randint(max_offset)
            data = logmel[:, :, offset:(input_length + offset)]
        else:
            if input_length > logmel.shape[2]:
                max_offset = input_length - logmel.shape[2]
                offset = np.random.randint(max_offset)
            else:
                # input_length = logmel.shape[2]
                offset = 0
            data = np.pad(logmel, ((0, 0), (0, 0), (offset, input_length - logmel.shape[2] - offset)), "constant")

        return data


class FAT_TestSet_logmel(Dataset):
    def __init__(self, config, fnames, transform=None, tta=5):
        self.config = config
        self.fnames = fnames
        self.transform = transform
        self.tta = tta

    def __len__(self):
        return len(self.fnames) * self.tta

    def __getitem__(self, idx):
        new_idx = idx % len(self.fnames)

        fe_dir = pathlib.Path(self.config['fe_dir'])
        filename = os.path.splitext(self.fnames[new_idx])[0] + '.pkl'
        file_path = fe_dir / filename

        # Read and Resample the audio
        data = self._random_selection(file_path)
        fname = self.fnames[new_idx]

        if self.transform is not None:
            data = self.transform(data)

        return data, fname

    def _random_selection(self, file_path):
        input_length = self.config['model']['params']['input_length']

        # Read the logmel pkl
        logmel = load_data(file_path)

        # standarize([0, 1])
        for i_channel in range(logmel.shape[0]):
            logmel[i_channel, :, :] = standarize(logmel[i_channel, :, :])

        # Random offset / Padding
        if logmel.shape[2] > input_length:   # logmel.shape = (3, n_mel, xx)
            max_offset = logmel.shape[2] - input_length
            offset = np.random.randint(max_offset)
            data = logmel[:, :, offset:(input_length + offset)]
        else:
            if input_length > logmel.shape[2]:
                max_offset = input_length - logmel.shape[2]
                offset = np.random.randint(max_offset)
            else:
                # input_length = logmel.shape[2]
                offset = 0
            data = np.pad(logmel, ((0, 0), (0, 0), (offset, input_length - logmel.shape[2] - offset)), "constant")

        return data
