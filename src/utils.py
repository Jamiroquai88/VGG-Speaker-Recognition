#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import time

import h5py
import numpy as np
import kaldi_io


def load_data(path, mode='train', spec_len=250):
    """

    Args:
        path (str): path to features in h5 format
        mode (str): `train` or `eval`
        spec_len (int): length of signal

    Returns:
        np.array: loaded features
    """
    utt, ark, position = path
    mat = list(kaldi_io.read_mat_ark(ark, offset=position))[0]
    segments = mat.transpose().copy()

    assert len(segments.shape) == 2, 'Segment `{}` for path `{}` does not have 2 dimensions.'.format(utt, ark)

    signal_len = segments.shape[1]
    if mode == 'train':
        randtime = np.random.randint(0, signal_len - spec_len)
        segments = segments[:, randtime:randtime + spec_len]
    return segments


def is_clean(key):
    return not key.endswith('-babble') and not key.endswith('-noise') and \
           not key.endswith('-music') and not key.endswith('-reverb')


def save_h5(path, mat):
    try:
        with h5py.File(path, 'w') as f:
            f.create_dataset(name='segments', data=mat)
    except IOError:
        time.sleep(1)
