#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: test_accuracy.py
@Time: 19-6-19 下午5:17
@Overview:
"""
# from __future__ import print_function
import argparse
import pdb
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os

import numpy as np
from tqdm import tqdm
from Define_Model.ResNet import SimpleResNet, ResNet
from Define_Model.TDNN import Time_Delay
from Define_Model.model import ResSpeakerModel
from Process_Data.KaldiDataset import KaldiTestDataset
from eval_metrics import evaluate_kaldi_eer
# from DeepSpeakerDataset_static import DeepSpeakerDataset
from Process_Data.DeepSpeakerDataset_dynamic import DeepSpeakerDataset, ClassificationDataset
from Process_Data.VoxcelebTestset import VoxcelebTestset
from Process_Data.voxceleb_wav_reader import wav_list_reader

from Define_Model.model import PairwiseDistance, ResCNNSpeaker, SuperficialResCNN
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, truncatedinputfromMFB, read_MFB, read_audio, \
    mk_MFB, concateinputfromMFB, varLengthFeat
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
parser.add_argument('--test-dir', type=str,
                    default='/home/cca01/work2019/yangwenhao/NewAdded/kaldi/egs/voxceleb/v4/data/voxceleb1_test_no_sil',
                    help='path to voxceleb1 test dataset')

parser.add_argument('--test-pairs-path', type=str, default='Data/dataset/voxceleb1/test_trials/ver_list.txt',
                    help='path to pairs file')

parser.add_argument('--resume', default='Data/checkpoint/SiResNet34/soft/kaldi/checkpoint_{}.pth', type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Training options
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--resnet-size', type=int, default=10, metavar='E',
                    help='depth of resnet to train (default: 34)')
parser.add_argument('--batch-size', type=int, default=512, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=3, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=1, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument('--acoustic-feature', choices=['fbank', 'spectrogram', 'mfcc'], default='fbank',
                    help='choose the acoustic features type.')

args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)

if args.cuda:
    cudnn.benchmark = True

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else PairwiseDistance(2)

# voxceleb, voxceleb_dev = wav_list_reader(args.dataroot)

transform = transforms.Compose([
    varLengthFeat(),
    totensor()
])

test_dir = KaldiTestDataset(dir=args.test_dir, transform=transform)
indices = list(range(len(test_dir)))
random.shuffle(indices)
indices = indices[:4800]
test_part = torch.utils.data.Subset(test_dir, indices)

writer = SummaryWriter('Data/checkpoint/SiResNet34/soft/kaldi/', filename_suffix='eer')


def main():
    # print the experiment configuration
    print('Parsed options:\n{}\n'.format(vars(args)))
    model = SimpleResNet(layers=[3, 4, 6, 3], num_classes=1211)

    if args.cuda:
        model.cuda()

    test_loader = torch.utils.data.DataLoader(test_part, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    epochs = np.arange(1, 29)

    for epoch in epochs:
        # Load model from Checkpoint file
        if os.path.isfile(args.resume.format(epoch)):
            print('=> loading checkpoint {}'.format(args.resume.format(epoch)))
            checkpoint = torch.load(args.resume.format(epoch))
            args.start_epoch = checkpoint['epoch']
            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
            model.load_state_dict(filtered)
            # optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
            break
        # train_test(train_loader, model, epoch)
        test(test_loader, model, epoch)

    writer.close()


def train_test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []
    x_vectors = []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, label) in pbar:

        if args.cuda:
            data_a = data_a.cuda()

        data_a = Variable(data_a)

        labels.append(label.data.numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    try:
        x_vectors = np.array(x_vectors)
        labels = np.array(labels)
        # labels.append(label.numpy())
    except Exception:
        pdb.set_trace()

    # err, accuracy= evaluate_eer(distances,labels)
    # eer, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim)
    try:
        # np.save('Data/xvector/train/x_vectors.npy', x_vectors)
        np.save('Data/xvector/train/label.npy', labels)
        print('Extracted {} x_vectors from train set.'.format(len(x_vectors)))
    except:
        pdb.set_trace()

    # tpr, fpr, accuracy, val, far = evaluate(distances, labels)

    # logger.log_value('Test Accuracy', np.mean(accuracy))


def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()

        data_a, data_p, label = Variable(data_a), \
                                Variable(data_p), Variable(label)

        # SiResNet34
        out_a = model.pre_forward_norm(data_a)
        out_p = model.pre_forward_norm(data_p)

        dists = l2_dist.forward(out_a, out_p)
        dists = dists.data.cpu().numpy()
        distances.append(dists)
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    writer.add_scalar('Test/EER', eer, epoch - 1)
    writer.add_scalar('Test/Threshold', eer_threshold, epoch - 1)

    print('\33[91mFor {}_distance, Test ERR: {:.8f}, Threshold: {:.8f}.\n\33[0m'.format('cos' if args.cos_sim else 'l2',
                                                                                        100. * eer, eer_threshold))


if __name__ == '__main__':
    main()
