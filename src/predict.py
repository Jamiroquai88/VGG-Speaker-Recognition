from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np

sys.path.append('../tool')
import toolkits
import utils as ut

import kaldi_io

import pdb
# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='', required=True, type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--kaldi-data-dirs', required=True, nargs='+', help='path to kaldi data directories')
parser.add_argument('--emb-out-dirs', required=True, nargs='+', help='output directories for storing embeddings')
parser.add_argument('--data_path', default='/media/weidi/2TB-2/datasets/voxceleb1/wav', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=10, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)

global args
args = parser.parse_args()

assert len(args.kaldi_data_dirs) == len(args.emb_out_dirs)

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model

    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    params = {'dim': (23, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)

    utt2ark, utt2idx, all_list, utt2data = {}, {}, [], {}
    for idx, kaldi_data_dir in enumerate(args.kaldi_data_dirs):
        if not os.path.exists(args.emb_out_dirs[idx]):
            os.makedirs(args.emb_out_dirs[idx])
        feats_path = os.path.join(kaldi_data_dir, 'feats.scp')
        vad_path = os.path.join(kaldi_data_dir, 'vad.scp')
        assert os.path.exists(feats_path), 'Path `{}` does not exists.'.format(feats_path)

        with open(feats_path) as f:
            for line in f:
                key, ark = line.split()
                ark, position = ark.split(':')
                input_tuple = (key, ark, int(position))
                utt2data[key] = ut.load_data(input_tuple, mode='eval')
                utt2idx[key] = idx

        with open(vad_path) as f:
            for line in f:
                key, ark = line.split()
                ark, position = ark.split(':')
                vad_array = None
                for ark_key, vec in kaldi_io.read_vec_flt_ark(ark):
                    if key == ark_key:
                        vad_array = np.array(vec, dtype=bool)
                assert vad_array is not None

                assert vad_array.size == utt2data[key].shape[1], 'Shapes does not fit: vad {}, mfcc {}'.format(
                    vad_array.size, utt2data[key].shape[1])
                utt2data[key] = ut.apply_cmvn_sliding(utt2data[key]).T[vad_array]

    # ==> load pre-trained model ???
    if os.path.isfile(args.resume):
        network_eval.load_weights(os.path.join(args.resume), by_name=True)
        print('==> successfully loaded model {}.'.format(args.resume))
    else:
        raise IOError("==> no checkpoint found at '{}'".format(args.resume))

    print('==> start testing.')

    # The feature extraction process has to be done sample-by-sample,
    # because each sample is of different lengths.
    for idx, utt in enumerate(utt2data):
        embedding = network_eval.predict(utt2data[utt].T[np.newaxis, :, :, np.newaxis]).squeeze()
        ut.write_txt_vectors(
            os.path.join(args.emb_out_dirs[utt2idx[utt]], 'xvector.{}.txt'.format(idx)), {utt: embedding})


if __name__ == "__main__":
    main()
