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


def load_data(path, mode='train', spec_len=250):
    """

    Args:
        path (str): path to features in h5 format
        mode (str): `train` or `eval`
        spec_len (int): length of signal

    Returns:
        np.array: loaded features
    """
    fea_path = path.replace('.wav', '.seg.h5')
    assert os.path.isfile(fea_path)
    # here, we need to avoid parallel reading of the same file
    segments = None
    for i in range(10):
        try:
            with h5py.File(fea_path) as f:
                segments = np.array(f['segments']).transpose().copy()
                break
        except IOError:
            time.sleep(0.1)
    assert segments is not None, 'Invalid path to h5 file `{}`.'.format(segments)

    assert len(segments.shape) == 2
    signal_len = segments.shape[1]
    if mode == 'train':
        randtime = np.random.randint(0, signal_len - spec_len)
        segments = segments[:, randtime:randtime + spec_len]
    return segments

