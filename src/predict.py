from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np
import keras2onnx
import onnx
import onnxruntime

sys.path.append('../tool')
import toolkits
import utils as ut

import kaldi_io

import pdb
# ===========================================
#        Parse the argument
# ===========================================
import argparse


XVEC_PER_ARK = 100


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
    params = {'dim': (30, None, 1),
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

    # ==> load pre-trained model ???
    if os.path.isfile(args.resume):
        network_eval.load_weights(os.path.join(args.resume), by_name=True)
        print('==> successfully loaded model {}.'.format(args.resume))
    else:
        raise IOError("==> no checkpoint found at '{}'".format(args.resume))
    network_eval.summary()    
    onnx_model = keras2onnx.convert_keras(network_eval, network_eval.name)
    onnx.save_model(onnx_model, f'{os.path.splitext(args.resume)[0]}.onnx')
    content = onnx_model.SerializeToString()
    onnx_sess = onnxruntime.InferenceSession(f'{os.path.splitext(args.resume)[0]}.onnx')
    onnx_input_name = onnx_sess.get_inputs()[0].name
    onnx_label_name = onnx_sess.get_outputs()[0].name

    for data_dir_idx, kaldi_data_dir in enumerate(args.kaldi_data_dirs):
        if not os.path.exists(args.emb_out_dirs[data_dir_idx]):
            os.makedirs(args.emb_out_dirs[data_dir_idx])
        feats_path = os.path.join(kaldi_data_dir, 'feats.scp')
        vad_path = os.path.join(kaldi_data_dir, 'vad.scp')
        assert os.path.exists(feats_path), 'Path `{}` does not exists.'.format(feats_path)
        
        vad_dict = None
        if os.path.exists(vad_path):
            print('Loading VAD ...')
            vad_dict, processed_vad_arks = {}, []
            with open(vad_path) as f:
                lines = f.readlines()
                num_lines = len(lines)
                for idx, line in enumerate(lines):
                    if idx % 100 == 99:
                        break
                        print(f'Loaded {idx}/{num_lines}.')
                    key, ark = line.split()
                    ark, position = ark.split(':')
                    if ark not in processed_vad_arks:
                        for ark_key, vec in kaldi_io.read_vec_flt_ark(ark):
                            vad_dict[ark_key] = np.array(vec, dtype=bool)

        print('Generating embeddings ...')
        emb_dict = {}
        with open(feats_path) as f:
            lines = f.readlines()
            num_lines = len(lines)
            for idx, line in enumerate(lines):
                if idx % XVEC_PER_ARK == 0:
                    xvec_ark_idx = idx / XVEC_PER_ARK
                    print(f'Processed {idx}/{num_lines}')
                    if len(emb_dict) > 0:
                        ut.write_txt_vectors(
                            os.path.join(args.emb_out_dirs[data_dir_idx], 'xvector.{}.txt'.format(xvec_ark_idx)), emb_dict)
                    emb_dict = {}
                key, ark = line.split()
                ark, position = ark.split(':')
                input_tuple = (key, ark, int(position))
                fea = ut.load_data(input_tuple, mode='eval')
                if vad_dict is not None:
                    vad = vad_dict[key]
                    assert vad.size == fea.shape[1], 'Shapes does not fit: vad {}, mfcc {}'.format(
                        vad.size, fea.shape[1])
                    fea = ut.apply_cmvn_sliding(fea.T)[vad]
                else:
                    fea = fea.T
                # cut recording which have over 2 minutes
                fea = fea[:12000, :]
                embedding = onnx_sess.run([onnx_label_name], {onnx_input_name: fea.T[np.newaxis, :, :, np.newaxis]})
                assert len(embedding) == 1
                embedding = embedding[0].squeeze() 
                emb_dict[key] = embedding

            if len(emb_dict) > 0:
                ut.write_txt_vectors(
                    os.path.join(args.emb_out_dirs[data_dir_idx], 'xvector.{}.txt'.format(xvec_ark_idx)), emb_dict)


if __name__ == "__main__":
    main()
