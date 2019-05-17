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


def load_data(path, mode='train', spec_len=250, tmp_dir='../data'):
    """

    Args:
        path (str): path to features in h5 format
        mode (str): `train` or `eval`
        spec_len (int): length of signal
        tmp_dir (str): temporary directory for faster loading
        use_clean_only (bool): use only clean version of inputs

    Returns:
        np.array: loaded features
    """
    utt, ark, position = path
    # h5_path = os.path.join(tmp_dir, '{}.h5'.format(utt))

    segments = None
    # if os.path.exists(h5_path):
    #     # print('file exists', h5_path)
    #     # avoid reading of the same file by another process
    #     for i in range(10):
    #         try:
    #             with h5py.File(h5_path) as f:
    #                 # print('success reading file', h5_path)
    #                 segments = np.array(f['segments'])
    #                 break
    #         except IOError:
    #             print('File `{}` was read by multiple processes at the same time.'.format(h5_path))
    #             time.sleep(0.1)
    # else:
        # print('file does not exists', h5_path)
        # print(list(kaldi_io.read_mat_ark(ark, offset=position))[0].shape)
    mat = list(kaldi_io.read_mat_ark(ark, offset=position))[0]
            # if use_clean_only:
            #     if not is_clean(key):
            #         continue
        # h5_path = os.path.join(tmp_dir, '{}.h5'.format(utt))
            # if key == utt:
    segments = mat.transpose().copy()
        # save_h5(h5_path, segments)
                # break

            # if not os.path.exists(h5_path):
            #     save_h5(h5_path, mat.transpose().copy())

    assert segments is not None, 'Segment `{}` for path `{}` is None.'.format(utt, h5_path)
    assert len(segments.shape) == 2, 'Segment `{}` for path `{}` does not have 2 dimensions.'.format(utt, h5_path)

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
