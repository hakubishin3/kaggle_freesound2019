import os
import librosa
import numpy as np
from joblib import Parallel, delayed
from .utils import save_data


def standarize(X, norm_max=None, norm_min=None, eps=1e-6):
    """ X is 2d-array. X.shape = (xx, yy)
    """
    # Standardize
    _min, _max = X.min(), X.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min

    if (_max - _min) > eps:
        # Scale to [0, 255]
        V = X
        V[V <= norm_min] = norm_min
        V[V >= norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def read_audio(wavefile: str, sampling_rate: int, min_data_length: int):
    # load wave data
    y, _ = librosa.load(wavefile, sr=sampling_rate)

    if len(y) > 0:
        # trim leading and trailing silence from an audio signal.
        y, _ = librosa.effects.trim(y)   # trim,top_db=60 (default)

        if len(y) < min_data_length:
            # delta features cant calcurated when logmel.shape[1] < 9
            padding = min_data_length - len(y)
            offset = padding // 2
            y = np.pad(y, (offset, min_data_length - len(y) - offset), 'constant')

    return y


def wav_to_logmel(wavelist, config, fe_dir):
    _ = Parallel(n_jobs=-1, verbose=1)(
        [delayed(tsfm_logmel)(wavefile, config, fe_dir) for wavefile in wavelist]
    )


def tsfm_logmel(wavefile, config, fe_dir, suffix=''):
    # get params
    params = config['features']['params']
    sampling_rate = params['sampling_rate']
    duration = params['duration']
    hop_length = params['factor__hop_length'] * duration
    n_mels = params['n_mels']
    n_fft = params['factor__n_fft'] * n_mels
    fmin = params['fmin']
    fmax = sampling_rate // params['factor__fmax']
    samples = sampling_rate * duration

    # get wav-data
    min_data_length = samples
    data = read_audio(wavefile, sampling_rate, min_data_length)

    # calc logmel
    if len(data) == 0:
        # If file is empty, fill logmel with 0.
        print("empty file: ", file_path)
        logmel = np.zeros((n_mels, n_mels), dtype=np.uint8)
        feats = np.stack([logmel, logmel, logmel], axis=-1)   # (n_mels, xx, 3)
    else:
        melspec = librosa.feature.melspectrogram(
            data, sr=sampling_rate,
            n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        logmel = librosa.core.power_to_db(melspec).astype(np.float32)
        delta = librosa.feature.delta(logmel).astype(np.float32)
        accelerate = librosa.feature.delta(logmel, order=2).astype(np.float32)

        # standarization
        #logmel = standarize(logmel)
        #delta = standarize(delta)
        #accelerate = standarize(accelerate)
        feats = np.stack([logmel, delta, accelerate], axis=-1)   # (n_mels, xx, 3)

    # save
    p_name = fe_dir / (os.path.splitext(os.path.basename(wavefile))[0] + suffix + '.pkl')
    save_data(p_name, feats)
