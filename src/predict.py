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
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--kaldi-data-dirs', required=True, nargs='+', help='path to kaldi data directories')
parser.add_argument('--emb-out-dirs', required=True, nargs='+', help='output directories for storing embeddings')
parser.add_argument('--data-path', default='/media/weidi/2TB-2/datasets/voxceleb1/wav', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost-cluster', default=2, type=int)
parser.add_argument('--vlad-cluster', default=10, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--num-dim', default=257, type=int, help='dimensionality of the features')
parser.add_argument('--max-spec-len', default=30000, type=int, help='maximal length of spectra')

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
    params = {'dim': (args.num_dim, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 8000,
              'batch_size': args.batch_size,
              'normalize': True,
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'], num_class=params['n_classes'],
                                                mode='eval', args=args)

    # ==> load pre-trained model ???
    if os.path.isfile(args.resume):
        network_eval.load_weights(os.path.join(args.resume), by_name=True)
        print('==> successfully loaded model {}.'.format(args.resume))
    else:
        raise IOError("==> no checkpoint found at '{}'".format(args.resume))

    print('==> start testing.')

    # The feature extraction process has to be done sample-by-sample,
    # because each sample is of different lengths.
    utt2ark, all_list, utt2wav, utt2lab, utt2emb = {}, [], {}, {}, {}
    for idx, kaldi_data_dir in enumerate(args.kaldi_data_dirs):
        if not os.path.exists(args.emb_out_dirs[idx]):
            os.makedirs(args.emb_out_dirs[idx])
        wav_scp_path = os.path.join(kaldi_data_dir, 'wav.scp')
        vad_scp_path = os.path.join(kaldi_data_dir, 'vad.scp')
        assert os.path.exists(wav_scp_path), 'Path `{}` does not exists.'.format(wav_scp_path)

        if os.path.exists(vad_scp_path):
            with open(vad_scp_path) as f:
                for line in f:
                    key, lab = line.split()
                    utt2lab[key] = lab

        with open(wav_scp_path) as f:
            for utt_idx, line in enumerate(f):
                splitted_line = line.split()
                key, wav = splitted_line[0], splitted_line[1:]
                utt2wav[key] = (key, wav)

                spec = ut.load_data(
                    path=utt2wav[key], vad_file=utt2lab.get(key, None), mode='eval')[:, :args.max_spec_len]
                spec = np.expand_dims(np.expand_dims(spec, 0), -1)
                embedding = network_eval.predict(spec).squeeze()
                utt2emb[key] = embedding

        ut.write_txt_vectors(os.path.join(args.emb_out_dirs[idx], 'xvector.0.txt'), utt2emb)


if __name__ == "__main__":
    main()
