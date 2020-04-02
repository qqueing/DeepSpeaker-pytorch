#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: test_libri.py
@Time: 2020/3/30 4:11 PM
@Overview:
"""
from __future__ import print_function
import argparse
import pathlib
import pdb
import random
import time

from kaldi_io import read_mat
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from Define_Model.ResNet import SimpleResNet, ExporingResNet
from Process_Data import constants as c
from Define_Model.SoftmaxLoss import AngleSoftmaxLoss
from Process_Data.KaldiDataset import ScriptTrainDataset, ScriptTestDataset, ScriptValidDataset, SitwTestDataset
from TrainAndTest.common_func import create_optimizer
from eval_metrics import evaluate_kaldi_eer
from Define_Model.model import PairwiseDistance, SuperficialResCNN
from Process_Data.audio_processing import concateinputfromMFB, PadCollate, varLengthFeat, to2tensor
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, read_MFB, read_audio
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
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Model options
# parser.add_argument('--train-dir', type=str,
#                     default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_aug_spect/dev',
#                     help='path to dataset')

parser.add_argument('--libri-dev-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/dev_kaldi',
                    help='path to librispeech test dataset')
parser.add_argument('--libri-test-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/test_kaldi',
                    help='path to librispeech test dataset')

# parser.add_argument('--check-path', default='Data/checkpoint/SiResNet34/soft/aug',
#                     help='folder to output model checkpoints')
parser.add_argument('--check-path', default='Data/checkpoint/ExResNet34/soft/dnn_cmvn_80',
                    help='folder to output model checkpoints')
parser.add_argument('--epochs', type=int, default=40, metavar='E',
                    help='number of epochs to train (default: 10)')

# Training options
parser.add_argument('--feat-dim', default=64, type=int, metavar='N',
                    help='acoustic feature dimension')
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=128, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')

parser.add_argument('--test-pairs', type=int, default=38400, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-input-per-file', type=int, default=4, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='BST',
                    help='input batch size for testing (default: 64)')

# parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
# parser.add_argument('--margin', type=float, default=3, metavar='MARGIN',
#                     help='the margin value for the angualr softmax loss function (default: 3.0')
# parser.add_argument('--loss-ratio', type=float, default=2.0, metavar='LOSSRATIO',
#                     help='the ratio softmax loss - triplet loss (default: 2.0')
# args for a-softmax
# parser.add_argument('--lambda-min', type=int, default=5, metavar='S',
#                     help='random seed (default: 0)')
# parser.add_argument('--lambda-max', type=int, default=1000, metavar='S',
#                     help='random seed (default: 0)')
# parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
#                     help='learning rate (default: 0.125)')
# parser.add_argument('--lr-decay', default=0, type=float, metavar='LRD',
#                     help='learning rate decay ratio (default: 1e-4')
# parser.add_argument('--weight-decay', default=5e-4, type=float,
#                     metavar='W', help='weight decay (default: 0.0)')
# parser.add_argument('--momentum', default=0.9, type=float,
#                     metavar='W', help='momentum for sgd (default: 0.9)')
# parser.add_argument('--dampening', default=0, type=float,
#                     metavar='W', help='dampening for sgd (default: 0.0)')
# parser.add_argument('--optimizer', default='sgd', type=str,
#                     metavar='OPT', help='The optimizer to use (default: Adagrad)')

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
random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    cudnn.benchmark = True

writer = SummaryWriter(logdir=args.check_path, filename_suffix='_librispeech')

kwargs = {'num_workers': 12, 'pin_memory': True} if args.cuda else {}
assert os.path.exists(args.check_path)

l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else PairwiseDistance(2)

if args.acoustic_feature == 'fbank':
    transform = transforms.Compose([
        concateinputfromMFB(num_frames=c.MINIMUIN_LENGTH, remove_vad=True),
        # varLengthFeat(),
        to2tensor()
    ])
    transform_T = transforms.Compose([
        concateinputfromMFB(num_frames=c.MINIMUIN_LENGTH, input_per_file=args.test_input_per_file, remove_vad=True),
        # varLengthFeat(),
        to2tensor()
    ])
else:
    transform = transforms.Compose([
        truncatedinput(),
        toMFB(),
        totensor(),
        # tonormal()
    ])

file_loader = read_mat


# pdb.set_trace()
# There are 721788 pairs in sitw eval Dataset.
# There are 338226 pairs in sitw dev Dataset.


def dev_test(sitw_dev_loader, sitw_test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()
    labels, distances = [], []
    pbar = tqdm(enumerate(sitw_dev_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        vec_shape = data_a.shape
        # pdb.set_trace()
        data_a = data_a.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])
        data_p = data_p.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a), Variable(data_p), Variable(label)

        # compute output
        out_a = model.pre_forward_norm(data_a)
        out_p = model.pre_forward_norm(data_p)

        dists = l2_dist.forward(out_a, out_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
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
    distances = np.nan_to_num(distances)

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
        out_a = model.pre_forward_norm(data_a)
        out_p = model.pre_forward_norm(data_p)

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
    distances = np.nan_to_num(distances)

    eer_t, eer_threshold_t, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    writer.add_scalars('Test/EER',
                       {'libirispeech_dev': 100. * eer_d, 'libirispeech_test': 100. * eer_t},
                       epoch)
    writer.add_scalars('Test/Threshold',
                       {'libirispeech_dev': eer_threshold_d, 'libirispeech_test': eer_threshold_t},
                       epoch)

    print('\n\33[91mFor libirispeech Dev ERR is {:.4f}%, Threshold is {},' \
          'Test ERR is {:.4f}%, Threshold is {}.\n\33[0m'.format(100. * eer_d, eer_threshold_d, 100. * eer_t,
                                                                 eer_threshold_t))
    torch.cuda.empty_cache()

def main():
    # print the experiment configuration
    num_spks = 1211
    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))
    print('Number of Speakers: {}.\n'.format(num_spks))

    # instantiate model and initialize weights
    # model = SuperficialResCNN(layers=[1, 1, 1, 0], embedding_size=args.embedding_size,
    #                           n_classes=num_spks, m=args.margin)
    # model = SimpleResNet(layers=[3, 4, 6, 3], num_classes=1211)
    model = ExporingResNet(layers=[3, 4, 6, 3], num_classes=1211)

    if args.cuda:
        model.cuda()

    # Datasets
    dev_dir = ScriptTestDataset(dir=args.libri_dev_dir, transform=transform_T, loader=file_loader)
    indices = list(range(len(dev_dir)))
    random.shuffle(indices)
    indices = indices[:args.test_pairs]
    dev_part = torch.utils.data.Subset(dev_dir, indices)

    test_dir = ScriptTestDataset(dir=args.libri_test_dir, transform=transform_T, loader=file_loader)
    indices = list(range(len(test_dir)))
    random.shuffle(indices)
    indices = indices[:args.test_pairs]
    test_part = torch.utils.data.Subset(test_dir, indices)

    libirispeech_test_loader = torch.utils.data.DataLoader(test_part, batch_size=args.test_batch_size, shuffle=False,
                                                           **kwargs)
    libirispeech_dev_loader = torch.utils.data.DataLoader(dev_part, batch_size=args.test_batch_size, shuffle=False,
                                                          **kwargs)
    epochs = np.arange(0, args.epochs + 1)
    resume_path = args.check_path + '/checkpoint_{}.pth'
    # optionally resume from a checkpoint
    for epoch in epochs:
        # Load model from Checkpoint file
        if os.path.isfile(resume_path.format(epoch)):
            print('=> loading checkpoint {}'.format(resume_path.format(epoch)))
            checkpoint = torch.load(resume_path.format(epoch))
            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
            model.load_state_dict(filtered)
        else:
            print('=> no checkpoint found at %s' % resume_path.format(epoch))
            continue

        dev_test(libirispeech_dev_loader, libirispeech_test_loader, model, epoch)

    writer.close()


# python TrainAndTest/Spectrogram/train_surescnn10_kaldi.py > Log/SuResCNN10/spect_161/

if __name__ == '__main__':
    main()
