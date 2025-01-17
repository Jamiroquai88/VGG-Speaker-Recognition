from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import random
import sys

import keras
import numpy as np

sys.path.append('../tool')

from utils import is_clean
import toolkits

# ===========================================
#        Parse the argument
# ===========================================

random.seed(42)
np.random.seed(42)


parser = argparse.ArgumentParser()

# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--kaldi-data-dir', required=True, type=str, help='path to kaldi data directory')
parser.add_argument('--use-clean-only', required=False, default=False, action='store_true', help='use only clean data')
parser.add_argument('--validation-ratio', required=False, type=float, default=0.01,
                    help='ratio of validation data to all training data')
# parser.add_argument('--files-per-split', required=False, type=int, default=1000, help='number of files in tmp split')
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=10, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--epochs', default=56, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--warmup_ratio', default=0, type=float)
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str)
parser.add_argument('--ohem_level', default=0, type=int,
                    help='pick hard samples from (ohem_level * batch_size) proposals, must be > 1')
parser.add_argument('--num-dim', default=23, type=int, help='dimensionality of the features')

global args
args = parser.parse_args()


def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model
    import generator

    # ==================================
    #       Get Train/Val.
    # ==================================
    feats_path = os.path.join(args.kaldi_data_dir, 'feats.scp')
    utt2spk_path = os.path.join(args.kaldi_data_dir, 'utt2spk')
    assert os.path.exists(feats_path), 'Path `{}` does not exists.'.format(feats_path)
    assert os.path.exists(utt2spk_path), 'Path `{}` does not exists.'.format(utt2spk_path)

    utt2ark = {}
    with open(feats_path) as f:
        for line in f:
            key, ark = line.split()
            if args.use_clean_only:
                if not is_clean(key):
                    continue
            ark, position = ark.split(':')
            utt2ark[key] = (key, ark, int(position))

    label2count, utt2label, label2int, label2utts = {}, {}, {}, {}
    with open(utt2spk_path) as f:
        for line in f:
            utt, label = line.split()
            if args.use_clean_only:
                if not is_clean(utt):
                    continue
            if label not in label2int:
                label2int[label] = len(label2int)
            label = label2int[label]
            utt2label[utt] = label
            if label not in label2count:
                label2count[label] = 0
            label2count[label] += 1
            if label not in label2utts:
                label2utts[label] = []
            label2utts[label].append(utt2ark[utt])

    # balancing classes
    trnlist, vallist, trnlb, vallb = [], [], [], []
    max_utts = max(label2count.values())
    for label in label2utts:
        # print('Balancing', label)
        validation_thr = label2count[label] * args.validation_ratio
        random.shuffle(label2utts[label])
        utts_array = np.array(label2utts[label])
        random_indexes = np.random.randint(low=0, high=label2count[label] - 1, size=max_utts)
        trn_indexes = random_indexes[random_indexes > validation_thr]
        val_indexes = random_indexes[random_indexes <= validation_thr]
        # print(np.max(trn_indexes), np.min(trn_indexes), np.max(val_indexes), np.min(val_indexes))
        trnlist.extend([(x[0], x[1], int(x[2])) for x in utts_array[trn_indexes]])
        trnlb.extend([label for x in range(len(trnlist))])
        # print(trnlist[:10], trnlb[:10])
        # 1/0
        vallist.extend([(x[0], x[1], int(x[2])) for x in utts_array[val_indexes]])
        vallb.extend([label for x in range(len(vallist))])

    # print(all_list[:10])
    # print(label2int)
    # print(label2count)
    # 1/0

    # label2val_count, trnlist, vallist, trnlb, vallb = {}, [], [], [], []
    # for utt in all_list:
    #     label = utt2label[utt[0]]
    #     if label not in label2val_count:
    #         label2val_count[label] = 0
    #     if label2val_count[label] <= label2count[label] * args.validation_ratio:
    #         # use for validation
    #         vallist.append(utt)
    #         vallb.append(label)
    #         label2val_count[label] += 1
    #     else:
    #         # use for training
    #         trnlist.append(utt)
    #         trnlb.append(label)

    # trnlb = keras.utils.to_categorical(trnlb)
    # vallb = keras.utils.to_categorical(vallb)

    # construct the data generator.
    params = {
        'dim': (args.num_dim, 250, 1),
        'mp_pooler': toolkits.set_mp(processes=4 * len(args.gpu.split(',')) + 1),
        'nfft': 512,
        'spec_len': 250,
        'win_length': 400,
        'hop_length': 160,
        'n_classes': len(label2count),
        'sampling_rate': 16000,
        'batch_size': args.batch_size,
        'shuffle': True,
        'normalize': True,
        'use_clean_only': args.use_clean_only
    }

    # Datasets
    partition = {'train': trnlist, 'val': vallist}
    labels = {'train': np.array(trnlb), 'val': np.array(vallb)}

    # Generators
    trn_gen = generator.DataGenerator(partition['train'], labels['train'], **params)
    val_gen = generator.DataGenerator(partition['val'], labels['val'], **params)
    network = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                           num_class=params['n_classes'],
                                           mode='train', args=args)

    # ==> load pre-trained model ???
    mgpu = len(keras.backend.tensorflow_backend._get_available_gpus())
    if args.resume:
        if os.path.isfile(args.resume):
            if mgpu == 1: network.load_weights(os.path.join(args.resume))
            else: network.layers[mgpu + 1].load_weights(os.path.join(args.resume))
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

    print(network.summary())
    print('==> gpu {} is, training {} features, validating {} features, classes: 0-{} '
          'loss: {}, aggregation: {}, ohemlevel: {}'.format(args.gpu, len(partition['train']),
                                                            len(partition['val']), np.max(labels['train']),
                                                            args.loss, args.aggregation_mode, args.ohem_level))

    model_path, log_path = set_path(args)
    normal_lr = keras.callbacks.LearningRateScheduler(step_decay)
    tbcallbacks = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=False,
                                              update_freq=args.batch_size * 16)
    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(model_path, 'weights-{epoch:02d}-{acc:.3f}.h5'),
                                                 monitor='loss',
                                                 mode='min',
                                                 save_best_only=True),
                 normal_lr, tbcallbacks]

    if args.ohem_level > 1:     # online hard negative mining will be used
        candidate_steps = int(len(partition['train']) // args.batch_size)
        iters_per_epoch = int(len(partition['train']) // (args.ohem_level*args.batch_size))

        ohem_generator = generator.OHEM_generator(network,
                                                  trn_gen,
                                                  candidate_steps,
                                                  args.ohem_level,
                                                  args.batch_size,
                                                  params['dim'],
                                                  params['n_classes']
                                                  )

        A = ohem_generator.next()   # for some reason, I need to warm up the generator

        network.fit_generator(generator.OHEM_generator(network, trn_gen, iters_per_epoch,
                                                       args.ohem_level, args.batch_size,
                                                       params['dim'], params['n_classes']),
                              steps_per_epoch=iters_per_epoch,
                              epochs=args.epochs,
                              max_queue_size=10,
                              callbacks=callbacks,
                              use_multiprocessing=False,
                              workers=1,
                              verbose=1,
                              validation_data=val_gen,
                              validation_steps=int(len(vallist) // args.batch_size))

    else:
        network.fit_generator(trn_gen,
                              steps_per_epoch=int(len(partition['train'])//args.batch_size),
                              epochs=args.epochs,
                              max_queue_size=10,
                              callbacks=callbacks,
                              use_multiprocessing=False,
                              workers=1,
                              verbose=1,
                              validation_data=val_gen,
                              validation_steps=int(len(vallist) // args.batch_size))


def step_decay(epoch):
    '''
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every step epochs.
    '''
    half_epoch = args.epochs // 2
    stage1, stage2, stage3 = int(half_epoch * 0.5), int(half_epoch * 0.8), half_epoch
    stage4 = stage3 + stage1
    stage5 = stage4 + (stage2 - stage1)
    stage6 = args.epochs

    if args.warmup_ratio:
        milestone = [2, stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [args.warmup_ratio, 1.0, 0.1, 0.01, 1.0, 0.1, 0.01]
    else:
        milestone = [stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [1.0, 0.1, 0.01, 1.0, 0.1, 0.01]

    lr = 0.005
    init_lr = args.lr
    stage = len(milestone)
    for s in range(stage):
        if epoch < milestone[s]:
            lr = init_lr * gamma[s]
            break
    print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
    return np.float(lr)


def set_path(args):
    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    if args.aggregation_mode == 'avg':
        exp_path = os.path.join(args.aggregation_mode+'_{}'.format(args.loss),
                                '{0}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                'lr{args.lr}_bdim{args.bottleneck_dim}_ohemlevel{args.ohem_level}'.format(date, args=args))
    elif args.aggregation_mode == 'vlad':
        exp_path = os.path.join(args.aggregation_mode+'_{}'.format(args.loss),
                                '{0}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                'lr{args.lr}_vlad{args.vlad_cluster}_'
                                'bdim{args.bottleneck_dim}_ohemlevel{args.ohem_level}'.format(date, args=args))
    elif args.aggregation_mode == 'gvlad':
        exp_path = os.path.join(args.aggregation_mode+'_{}'.format(args.loss),
                                '{0}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                'lr{args.lr}_vlad{args.vlad_cluster}_ghost{args.ghost_cluster}_'
                                'bdim{args.bottleneck_dim}_ohemlevel{args.ohem_level}'.format(date, args=args))
    else:
        raise IOError('==> unknown aggregation mode.')
    model_path = os.path.join('../model', exp_path)
    log_path = os.path.join('../log', exp_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists(log_path): os.makedirs(log_path)
    return model_path, log_path


if __name__ == "__main__":
    main()
