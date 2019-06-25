# Third Party

import os
import tempfile
import subprocess

from collections import defaultdict, namedtuple
import librosa
import numpy as np
from scipy.io.wavfile import read
from six import iteritems


FS_LAB = 1e7    # sampling frequency used in .lab files
Segment = namedtuple('Segment', ['start', 'end', 'label', 'channel'])    # segment = one line of .lab file
Segment.__new__.__defaults__ = (0,)


# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(vid_path, sr, mode='train', spec_len=250, vad_file=None, rand_int=100):
    try:
        sample_rate, wav = read(vid_path, mmap=True)
    except ValueError:
        raise ValueError('Problem detected when loading wave file `{}`.'.format(vid_path))
    assert sample_rate == sr, 'Expected sample rate {} does not match sample rate of file `{}`.'.format(sr, vid_path)
    assert len(wav.shape) == 1, 'Expected only single channel in `{}`.'.format(vid_path)

    sig_len = wav.size / sample_rate * 100
    if mode == 'train':
        is_defined = False
        if vad_file is not None:
            vad_array = load_vad_as_bool_array(vad_file, wav.shape, sr)
            for i in range(rand_int):
                randtime = np.random.randint(0, sig_len - spec_len)
                start_frame = randtime * sample_rate / 100
                output_len = spec_len * sample_rate / 100
                # if there is at least half of the frames with speech
                vad_segment = vad_array[start_frame:start_frame + output_len]
                if vad_segment[vad_segment == True].size > output_len / 2:
                    is_defined = True
                    wav = wav[start_frame:start_frame + output_len]
                    break

        if not is_defined:
            randtime = np.random.randint(0, sig_len - spec_len)
            start_frame = randtime * sample_rate / 100
            output_len = spec_len * sample_rate / 100
            wav = wav[start_frame:start_frame + output_len]

    wav = librosa.util.buf_to_float(wav)
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav


def load_vad_as_bool_array(vad_file, wav_shape, sr):
    bool_array = np.ones(shape=wav_shape[0], dtype=np.bool)
    if os.path.isfile(vad_file):
        vad = read_lab_file(vad_file)
        assert len(vad) == 1, 'Got {} channels in file `{}`, expecting only 1.'.format(len(vad), vad_file)
        for segment in vad[0]:
            if segment.label is not True:
                start, end = int(segment.start * sr), int(segment.end * sr)
                if end >= wav_shape[0]:
                    end = wav_shape[0] - 1
                if 0 <= start <= end <= wav_shape[0]:
                    bool_array[start:end] = False
    return bool_array


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(path, win_length=400, sr=8000, hop_length=160, n_fft=512, spec_len=250, vad_file=None, mode='train'):
    key, cmd = path
    if len(cmd) == 1:
        # avoid creation of tmp file
        wav = load_wav(cmd[0], sr=sr, mode=mode, spec_len=spec_len, vad_file=vad_file)
    else:
        with tempfile.NamedTemporaryFile() as f:
            cmd = cmd[0:-1]
            subprocess.check_call(args=[' '.join(cmd)], stdout=f, shell=True)
            path = f.name
        wav = load_wav(path, sr=sr, mode=mode, spec_len=spec_len, vad_file=vad_file)

    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    spec_mag = mag.T[:, :spec_len]

    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
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
