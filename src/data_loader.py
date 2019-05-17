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


class FAT_TrainSet_logmel(Dataset):
    def __init__(self, config, dataframe, labels, transform=None):
        self.config = config
        self.dataframe = dataframe
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # get filename
        fe_dir = pathlib.Path(self.config['fe_dir'])
        filename = os.path.splitext(self.dataframe["fname"][idx])[0] + '.pkl'
        file_path = fe_dir / filename

        # Read and Resample the audio
        data = self._random_selection(file_path)
        label = torch.from_numpy(self.labels[idx]).float()

        if self.transform is not None:
            data = self.transform(data)

        # standarize
        data = data.div_(255)

        return data, label

    def _random_selection(self, file_path):
        input_length = self.config['model']['params']['input_length']

        # Read the logmel pkl
        logmel = load_data(file_path)

        # random offset and convert to image
        image = Image.fromarray(logmel, mode='RGB')
        time_dim, base_dim = image.size
        crop = random.randint(0, time_dim - base_dim)
        image = image.crop([crop, 0, crop + base_dim, base_dim])

        return image


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

        # standarize
        data = data.div_(255)

        return data, fname

    def _random_selection(self, file_path):
        input_length = self.config['model']['params']['input_length']

        # Read the logmel pkl
        logmel = load_data(file_path)

        # random offset and convert to image
        image = Image.fromarray(logmel, mode='RGB')
        time_dim, base_dim = image.size
        crop = random.randint(0, time_dim - base_dim)
        image = image.crop([crop, 0, crop + base_dim, base_dim])

        return image
