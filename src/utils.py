# Third Party

import os
import tempfile
import subprocess

import librosa
from six import iteritems
from collections import defaultdict, namedtuple
import numpy as np


FS_LAB = 1e7    # sampling frequency used in .lab files
Segment = namedtuple('Segment', ['start', 'end', 'label', 'channel'])    # segment = one line of .lab file
Segment.__new__.__defaults__ = (0,)


# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(vid_path, sr, mode='train'):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(path, win_length=400, sr=8000, hop_length=160, n_fft=512, spec_len=250, vad_dir=None, mode='train'):
    key, cmd = path
    if len(cmd) == 1:
        # avoid creation of tmp file
        wav = load_wav(cmd[0], sr=sr, mode=mode)
    else:
        with tempfile.NamedTemporaryFile() as f:
            cmd = cmd[0:-1]
            subprocess.check_call(args=[' '.join(cmd)], stdout=f, shell=True)
            path = f.name
        wav = load_wav(path, sr=sr, mode=mode)

    if vad_dir is not None:
        vad_file = os.path.join(vad_dir, '{}.lab'.format(key))
        if os.path.isfile(vad_file):
            vad = read_lab_file(vad_file)
            assert len(vad) == 1, 'Got {} channels in file `{}`, expecting only 1.'.format(len(vad), vad_file)
            bool_array = np.ones(shape=wav.shape[0], dtype=np.bool)
            for segment in vad[0]:
                if segment.label is not True:
                    start, end = int(segment.start * sr), int(segment.end * sr)
                    assert 0 <= start < end < wav.shape[0], \
                        'Start({}) in file `{}` not larger than end({}) with size {}.'.format(
                            start, vad_file, end, wav.shape[0])
                    bool_array[start:end] = False

            # print('Before', vad_file, wav.shape, bool_array[bool_array == True].shape)
            wav = wav[bool_array]
            # print('After', vad_file, wav.shape)

    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, time-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    # print(mu.shape, std.shape, spec_mag.shape)
    return (spec_mag - mu) / (std + 1e-5)


def is_clean(key):
    """

    Args:
        key:

    Returns:

    """
    return not key.endswith('-babble') and not key.endswith('-noise') and \
           not key.endswith('-music') and not key.endswith('-reverb')


def read_lab_file(path):
    """ Read contents of .lab file into labeled segments.

    Segment:
        - start (float): start time of the segment in seconds
        - end (float): end time of the segment in seconds
        - label (bool): True if voice, else False
        - channel (int): number of channel

    Args:
        path (str): path to .lab (segmentation) file

    Returns:
        List[List[Segment]]: list of labeled segments, for each channel one list
    """
    output = defaultdict(list)
    for segment in read_bsapi_lab_file(path):
        channel = segment.channel
        output[channel].append(Segment(segment.start, segment.end, segment.label == 'voice', channel))
    return [segments for channel, segments in sorted(iteritems(output))]


def read_bsapi_lab_file(path):
    """ Read contents of .lab file into labeled segments.

    Segment:
        - start (float): start time of the segment in seconds
        - end (float): end time of the segment in seconds
        - label (str): label of the segment
        - channel (int): number of channel

    Args:
        path (str): path to .lab (segmentation) file

    Yields:
        Segment: labeled segment
    """
    with open(path) as f:
        for line in f:
            line_split = line.split()
            start, end, label = line_split[:3]
            channel = int(line_split[3]) if len(line_split) > 3 else None
            yield Segment(float(start) / FS_LAB, float(end) / FS_LAB, label, channel)
