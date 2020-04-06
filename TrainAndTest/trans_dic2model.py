#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: trans_dic2model.py
@Time: 2020/4/6 1:38 PM
@Overview:
"""

from __future__ import print_function
import argparse
import pdb
import time
import torch
import os

from Define_Model.ResNet import LocalResNet
from Define_Model.SoftmaxLoss import AngleSoftmaxLoss, AngleLinear, AdditiveMarginLinear, AMSoftmaxLoss
# Version conflict
import warnings

warnings.filterwarnings("ignore")

import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

# Training settings
parser = argparse.ArgumentParser(description='Trans dict to model object')
# Model options
parser.add_argument('--check-path', default='Data/checkpoint/LoResNet10/spect/asoft',
                    help='folder to output model checkpoints')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=20, metavar='E',
                    help='number of epochs to train (default: 10)')

# Training options
parser.add_argument('--feat-dim', default=161, type=int, metavar='N',
                    help='acoustic feature dimension')
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='ES',
                    help='Dimensionality of the embedding')

# parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
parser.add_argument('--loss-type', type=str, default='asoft', choices=['soft', 'asoft', 'center', 'amsoft'],
                    help='path to voxceleb1 test dataset')
parser.add_argument('--m', type=float, default=3, metavar='M',
                    help='the margin value for the angualr softmax loss function (default: 3.0')
parser.add_argument('--margin', type=float, default=0.3, metavar='MARGIN',
                    help='the margin value for the angualr softmax loss function (default: 3.0')
parser.add_argument('--s', type=float, default=15, metavar='S',
                    help='the margin value for the angualr softmax loss function (default: 3.0')
parser.add_argument('--loss-ratio', type=float, default=0.1, metavar='LOSSRATIO',
                    help='the ratio softmax loss - triplet loss (default: 2.0')

# args for a-softmax
parser.add_argument('--lambda-min', type=int, default=5, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--lambda-max', type=int, default=12500, metavar='S',
                    help='random seed (default: 0)')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=123456, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=12, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument('--acoustic-feature', choices=['fbank', 'spectrogram', 'mfcc'], default='fbank',
                    help='choose the acoustic features type.')
parser.add_argument('--makemfb', action='store_true', default=False,
                    help='need to make mfb file')
parser.add_argument('--makespec', action='store_true', default=False,
                    help='need to make spectrograms file')

args = parser.parse_args()

# Set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # test_display_triplet_distance = False
    # print the experiment configuration
    num_spks = 1211
    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))
    print('Number of Speakers: {}.\n'.format(num_spks))

    # instantiate model and initialize weights
    model = LocalResNet(resnet_size=10, embedding_size=args.embedding_size, num_classes=num_spks)
    # start_epoch = 0
    if args.loss_type == 'asoft':
        model.classifier = AngleLinear(in_features=args.embedding_size, out_features=num_spks, m=args.m)

    elif args.loss_type == 'amsoft':
        model.classifier = AdditiveMarginLinear(feat_dim=args.embedding_size, n_classes=num_spks)

    # ['soft', 'asoft', 'center', 'amsoft'], optionally resume from a checkpoint
    start = 1
    print('Start epoch is : ' + str(start))
    # start = 0
    end = start + args.epochs

    for epoch in range(start, end):
        check_path = '{}/checkpoint_{}.pth'.format(args.check_path, epoch)
        if os.path.isfile(check_path):
            print('=> loading checkpoint {}'.format(check_path))
            checkpoint = torch.load(check_path)
            # pdb.set_trace()
            e = checkpoint['epoch']

            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
            model_dict = model.state_dict()
            model_dict.update(filtered)

            model.load_state_dict(model_dict)
            ce = checkpoint['criterion']

            torch.save({'epoch': e,
                        'model': model,
                        'criterion': ce},
                       check_path + '.new')

            print('=> Saving new checkpoint at {}'.format(check_path + '.new'))
        else:
            print('=> no checkpoint found at {}'.format(check_path))


# python TrainAndTest/Spectrogram/train_surescnn10_kaldi.py > Log/SuResCNN10/spect_161/

if __name__ == '__main__':
    main()
