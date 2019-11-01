#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import os
import time

import h5py
import numpy as np
import kaldi_io


def load_data(path, mode='train', spec_len=200):
    """

    Args:
        path (Tuple[str, str, int]): path to features in h5 format
        mode (str): `train` or `eval`
        spec_len (int): length of signal

    Returns:
        np.array: loaded features
    """
    utt, ark, position = path
    try:
        with open(ark, 'rb') as f:
            f.seek(position - len(utt) - 1)
            ark_key = kaldi_io.read_key(f)
            assert ark_key == utt, 'Keys does not match: `{}` and `{}`.'.format(ark_key, utt)
            mat = kaldi_io.read_mat(f)
            segments = mat.transpose().copy()

        assert len(segments.shape) == 2, 'Segment `{}` for path `{}` does not have 2 dimensions.'.format(utt, ark)

        signal_len = segments.shape[1]

        if mode == 'train':
            if signal_len <= spec_len:
                for i in range(spec_len / signal_len):
                    segments = np.concatenate((segments, segments), axis=1)
                signal_len = segments.shape[1]

            randtime = np.random.randint(0, signal_len - spec_len)
            segments = segments[:, randtime:randtime + spec_len]
        return segments
    except AssertionError:
        print('Problem with utterance', utt, ark, position)
        return np.zeros(shape=(30, spec_len))


def is_clean(key):
    return not key.endswith('-babble') and not key.endswith('-noise') and \
           not key.endswith('-music') and not key.endswith('-reverb')


def save_h5(path, mat):
    try:
        with h5py.File(path, 'w') as f:
            f.create_dataset(name='segments', data=mat)
    except IOError:
        time.sleep(1)


def write_txt_vectors(path, data_dict):
    """ Write vectors file in text format.

    Args:
        path (str): path to txt file
        data_dict: (Dict[np.array]): name to array mapping
    """
    with open(path, 'w') as f:
        for name in sorted(data_dict):
            f.write('{}  [ {} ]{}'.format(name, ' '.join(str(x) for x in data_dict[name]), os.linesep))


def apply_cmvn_sliding(matrix, window_size=300, bsapi_compat=False):
    """ Equivalent of kaldi's apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300.

    Borrowed from git@gitlab.int.phonexia.com:CORE-team/ASR-tools.git@31-snyder2bsapi:snydernet/extract_xvectors_main.py

    Args:
        matrix (np.ndarray): feature matrix of shape (n_samples, n_dim)
        window_size (int): length of the cmvn window in frames
        bsapi_compat (bool): be BSAPI compatible

    Returns:
        np.ndarray: processed feature matrix of the same shape
    """
    prev_t0 = 0
    prev_t1 = 0
    sum_val = np.zeros(shape=(matrix.shape[1],), dtype=np.float64)
    retval = np.zeros_like(matrix, dtype=np.float64)
    max_t = matrix.shape[0]
    for t in range(max_t):
        if bsapi_compat:
            t0 = max(0, t - window_size // 2)
            t1 = min(t + window_size // 2 + 1, max_t)
        else:
            t0 = max(0, t - window_size // 2)
            t1 = min(max_t, t0 + window_size)
            if t1 - t0 < window_size:
                t0 = max(0, t1 - window_size)

        sum_val -= np.sum(matrix[prev_t0:t0, :], axis=0)
        sum_val += np.sum(matrix[prev_t1:t1, :], axis=0)
        retval[t, :] = matrix[t, :] - sum_val / (t1 - t0)
        prev_t0 = t0
        prev_t1 = t1
    return retval
