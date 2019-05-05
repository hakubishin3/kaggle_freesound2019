import os
import librosa
import numpy as np
from joblib import Parallel, delayed
from .utils import save_data


def standarize(X, eps=1e-6):
    """ X is 2d-array. X.shape = (xx, yy)
    """
    max_ = X.max()
    min_ = X.min()
    if (max_ - min_) > eps:
        # normalize to [0, 255]
        # X = (X - min_) / (max_ - min_) * 255
        # normalize to [0, 1]
        X = (X - min_) / (max_ - min_)
    else:
        # just zero
        X = np.zeros_like(X)

    # X = X.astype(np.uint8)
    return X


def read_audio(wavefile: str, sampling_rate: int, min_data_length: int):
    # load wave data
    y, _ = librosa.load(wavefile, sr=sampling_rate)
    y = y.astype(np.float32)

    if len(y) > 0:
        # trim leading and trailing silence from an audio signal.
        y, _ = librosa.effects.trim(y)   # trim,top_db=60 (default)

        if len(y) < min_data_length:
            # delta features cant calcurated when logmel.shape[1] < 9
            padding = min_data_length - len(y)
            offset = padding // 2
            y = np.pad(y, (offset, min_data_length - len(y) - offset), 'constant')

    return y


def wav_to_logmel(wavelist, params, fe_dir):
    _ = Parallel(n_jobs=-1, verbose=1)(
        [delayed(tsfm_logmel)(wavefile, params, fe_dir) for wavefile in wavelist]
    )


def tsfm_logmel(wavefile, params, fe_dir):
    # get params
    sampling_rate = params['sampling_rate']
    duration = params['duration']
    hop_length = params['factor__hop_length'] * duration
    n_mels = params['n_mels']
    n_fft = params['factor__n_fft'] * n_mels
    fmin = params['fmin']
    fmax = sampling_rate // params['factor__fmax']
    samples = sampling_rate * duration

    # delta features can't calcurate when logmel.shape[1] < 9
    # min_data_length = int(9 * hop_length)
    min_data_length = samples

    # get wav-data
    data = read_audio(wavefile, params['sampling_rate'], min_data_length)

    # calc logmel
    if len(data) == 0:
        # If file is empty, fill logmel with 0.
        print("empty file: ", file_path)
        logmel = np.zeros((n_mels, n_mels))
        # feats = np.stack((logmel, logmel, logmel), axis=-1)   # (n_mels, xx, 3): because of convert PIL.Image later
        feats = np.stack((logmel, logmel, logmel))   # (3, n_mels, xx)
    else:
        melspec = librosa.feature.melspectrogram(
            data, sr=sampling_rate,
            n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        logmel = librosa.core.power_to_db(melspec).astype(np.float32)
        delta = librosa.feature.delta(logmel).astype(np.float32)
        accelerate = librosa.feature.delta(logmel, order=2).astype(np.float32)
        # feats = np.stack((logmel, delta, accelerate), axis=-1)   # (n_mels, xx, 3): because of convert PIL.Image later
        feats = np.stack((logmel, delta, accelerate))   # (3, n_mels, xx)

    # save
    p_name = fe_dir / (os.path.splitext(os.path.basename(wavefile))[0] + '.pkl')
    save_data(p_name, feats)
