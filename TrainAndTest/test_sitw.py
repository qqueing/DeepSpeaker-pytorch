#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: test_sitw.py
@Time: 2020/3/21 11:53 AM
@Overview:
"""

from __future__ import print_function
import argparse
import os.path as osp
import pdb
import random
import sys
import time
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
import numpy as np
from tqdm import tqdm
from kaldi_io import read_mat

from Define_Model.ResNet import LocalResNet
from Define_Model.SoftmaxLoss import AngleLinear, AdditiveMarginLinear
from Process_Data import constants as c
from Process_Data.KaldiDataset import ScriptTrainDataset, ScriptTestDataset, ScriptValidDataset, SitwTestDataset
from eval_metrics import evaluate_kaldi_eer
from Define_Model.model import PairwiseDistance, SuperficialResCNN
from Process_Data.audio_processing import concateinputfromMFB, PadCollate, varLengthFeat, to2tensor
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, read_audio
# Version conflict
import warnings

from logger import NewLogger

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
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
parser.add_argument('--sitw-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/sitw',
                    help='path to voxceleb1 test dataset')
parser.add_argument('--nj', default=8, type=int, metavar='NJ',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--check-path', default='Data/checkpoint/SuResCNN10/spect/kaldi_5wd',
                    help='folder to output model checkpoints')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=25, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--veri-pairs', type=int, default=18000, metavar='VP',
                    help='number of epochs to train (default: 10)')

# Training options
parser.add_argument('--loss-type', type=str, default='soft', choices=['soft', 'asoft', 'center', 'amsoft'],
                    help='path to voxceleb1 test dataset')

parser.add_argument('--feat-dim', default=161, type=int, metavar='N',
                    help='acoustic feature dimension')
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--test-input-per-file', type=int, default=4, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--m', type=float, default=3, metavar='M',
                    help='the margin value for the angualr softmax loss function (default: 3.0')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=123456, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
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

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    cudnn.benchmark = True

writer = SummaryWriter(logdir=args.check_path, filename_suffix='only_sitw')

kwargs = {'num_workers': 12, 'pin_memory': True} if args.cuda else {}
assert os.path.exists(args.check_path)
sys.stdout = NewLogger(osp.join(args.check_path, 'sitw.txt'))

l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else PairwiseDistance(2)

if args.acoustic_feature == 'fbank':
    transform = transforms.Compose([
        concateinputfromMFB(num_frames=c.NUM_FRAMES_SPECT, remove_vad=False),
        # varLengthFeat(),
        to2tensor()
    ])
    transform_T = transforms.Compose([
        concateinputfromMFB(num_frames=c.NUM_FRAMES_SPECT, input_per_file=args.test_input_per_file, remove_vad=False),
        # varLengthFeat(),
        to2tensor()
    ])
    file_loader = read_mat
else:
    transform = transforms.Compose([
        truncatedinput(),
        toMFB(),
        totensor(),
        # tonormal()
    ])
    file_loader = read_audio
# pdb.set_trace()

sitw_test_dir = SitwTestDataset(sitw_dir=args.sitw_dir, sitw_set='eval', transform=transform_T, set_suffix='')
if len(sitw_test_dir) < args.veri_pairs:
    args.veri_pairs = len(sitw_test_dir)
    print('There are %d verification pairs.' % len(sitw_test_dir))
else:
    sitw_test_dir.partition(args.veri_pairs)

sitw_dev_dir = SitwTestDataset(sitw_dir=args.sitw_dir, sitw_set='dev', transform=transform_T, set_suffix='')
if len(sitw_dev_dir) < args.veri_pairs:
    args.veri_pairs = len(sitw_dev_dir)
    print('There are %d verification pairs.' % len(sitw_dev_dir))
else:
    sitw_dev_dir.partition(args.veri_pairs)


def sitw_test(sitw_dev_loader, sitw_test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()
    labels, distances = [], []
    pbar = tqdm(enumerate(sitw_dev_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        vec_shape = data_a.shape
        # print(label)
        # pdb.set_trace()
        if vec_shape[1] != 1:
            data_a = data_a.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])
            data_p = data_p.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a), Variable(data_p), Variable(label)

        # compute output
        _, out_a_ = model(data_a)
        _, out_p_ = model(data_p)
        out_a = out_a_
        out_p = out_p_

        dists = l2_dist.forward(out_a, out_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        if vec_shape[1] != 1:
            dists = dists.reshape(vec_shape[0], vec_shape[1]).mean(axis=1)
        dists = dists.data.cpu().numpy()

        distances.append(dists)
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * vec_shape[0], len(sitw_dev_loader.dataset),
                       100. * batch_idx / len(sitw_dev_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    eer_d, eer_threshold_d, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)

    torch.cuda.empty_cache()

    labels, distances = [], []
    pbar = tqdm(enumerate(sitw_test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        vec_shape = data_a.shape
        # pdb.set_trace()
        data_a = data_a.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])
        data_p = data_p.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a), Variable(data_p), Variable(label)

        # compute output
        _, out_a_ = model(data_a)
        _, out_p_ = model(data_p)
        out_a = out_a_
        out_p = out_p_

        dists = l2_dist.forward(out_a, out_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        dists = dists.reshape(vec_shape[0], vec_shape[1]).mean(axis=1)
        dists = dists.data.cpu().numpy()

        distances.append(dists)
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * vec_shape[0], len(sitw_test_loader.dataset),
                       100. * batch_idx / len(sitw_test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    eer_t, eer_threshold_t, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    torch.cuda.empty_cache()

    writer.add_scalars('Test/EER',
                       {'sitw_dev': 100. * eer_d, 'sitw_test': 100. * eer_t},
                       epoch)

    writer.add_scalars('Test/Threshold',
                       {'sitw_dev': eer_threshold_d, 'sitw_test': eer_threshold_t},
                       epoch)

    print('\33[91mFor Sitw Dev ERR: {:.4f}%, Threshold: {},' \
          'Test ERR: {:.4f}%, Threshold: {}.\n\33[0m'.format(100. * eer_d, eer_threshold_d,
                                                             100. * eer_t, eer_threshold_t))


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # print the experiment configuration
    num_spks = 1211
    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))
    print('Number of Speakers: {}.\n'.format(num_spks))

    # instantiate model and initialize weights
    # model = SuperficialResCNN(layers=[1, 1, 1, 0], embedding_size=args.embedding_size,
    #                           n_classes=num_spks, m=args.margin)

    model = LocalResNet(resnet_size=10, embedding_size=args.embedding_size, num_classes=num_spks)

    if args.loss_type == 'asoft':
        model.classifier = AngleLinear(in_features=args.embedding_size, out_features=num_spks, m=args.m)
    elif args.loss_type == 'amsoft':
        model.classifier = AdditiveMarginLinear(feat_dim=args.embedding_size, n_classes=num_spks)

    if args.cuda:
        model.cuda()

    # optionally resume from a checkpoint
    sitw_test_loader = torch.utils.data.DataLoader(sitw_dev_dir, batch_size=args.test_batch_size,
                                                   shuffle=False, **kwargs)
    sitw_dev_loader = torch.utils.data.DataLoader(sitw_dev_dir, batch_size=args.test_batch_size,
                                                  shuffle=False, **kwargs)
    epochs = np.arange(1, args.epochs + 1)
    resume_path = args.check_path + '/checkpoint_{}.pth'
    for epoch in epochs:
        # Load model from Checkpoint file
        if os.path.isfile(resume_path.format(epoch)):
            print('=> loading checkpoint {}'.format(resume_path.format(epoch)))

            checkpoint = torch.load(resume_path.format(epoch))
            start_epoch = checkpoint['epoch']
            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
            model.load_state_dict(filtered)
        else:
            print('=> no checkpoint found at %s' % resume_path.format(epoch))
            continue

        sitw_test(sitw_dev_loader, sitw_test_loader, model, start_epoch)

    writer.close()


# python TrainAndTest/Spectrogram/train_surescnn10_kaldi.py > Log/SuResCNN10/spect_161/

if __name__ == '__main__':
    main()
