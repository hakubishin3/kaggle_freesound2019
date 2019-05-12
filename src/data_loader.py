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


def get_silent_wav_list():
    """https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/89108#latest-529449
    """
    silent_wav_list = [
        '1d44b0bd.wav',
        '02f274b2.wav',
        '08b34136.wav',
        '1af3bd88.wav',
        '1fd4f275.wav',
        '2f503375.wav',
        '3496256e.wav',
        '551a4b3b.wav',
        '5a5761c9.wav',
        '6d062e59.wav',
        '769d131d.wav',
        '8c712129.wav',
        '988cf8f2.wav',
        '9f4fa2df.wav',
        'b1d2590c.wav',
        'be273a3c.wav',
        'd527dcf0.wav',
        'e4faa2e1.wav',
        'fa659a71.wav',
        'fba392d8.wav'
    ]
    return silent_wav_list


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


class FAT_TrainSet_logmel(Dataset):
    def __init__(self, config, dataframe, labels, offline_aug, specAug, transform=None):
        self.config = config
        self.dataframe = dataframe
        self.labels = labels
        self.transform = transform
        self.offline_aug = offline_aug
        self.specAug = specAug

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # get augmentation filename if offline_aug is True
        if self.offline_aug:
            n_aug = self.config['offline-augment']['n_aug']
            i_aug = np.random.randint(n_aug)
            suffix = f'_aug{i_aug+1}'
        else:
            suffix = ''

        # get filename
        fe_dir = pathlib.Path(self.config['fe_dir'])
        filename = os.path.splitext(self.dataframe["fname"][idx])[0] + suffix + '.pkl'
        file_path = fe_dir / filename

        # Read and Resample the audio
        data = self._random_selection(file_path)
        label = torch.from_numpy(self.labels[idx]).float()

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
            # Random offset
            max_offset = logmel.shape[2] - input_length
            offset = np.random.randint(max_offset)
            data = logmel[:, :, offset:(input_length + offset)]
        else:
            # Padding
            if input_length > logmel.shape[2]:
                max_offset = input_length - logmel.shape[2]
                offset = np.random.randint(max_offset)
            else:
                # input_length = logmel.shape[2]
                offset = 0
            data = np.pad(logmel, ((0, 0), (0, 0), (offset, input_length - logmel.shape[2] - offset)), "constant")

        return data
