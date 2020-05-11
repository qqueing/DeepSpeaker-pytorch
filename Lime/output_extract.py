#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: output_extract.py
@Time: 2020/3/21 5:57 PM
@Overview:
"""
from __future__ import print_function

import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch
import torch._utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from kaldi_io import read_mat
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from Define_Model.SoftmaxLoss import AngleLinear, AdditiveMarginLinear
from Define_Model.model import PairwiseDistance
from Process_Data.KaldiDataset import ScriptTrainDataset, \
    ScriptTestDataset, ScriptValidDataset
from Process_Data.audio_processing import varLengthFeat, to2tensor, mvnormal, concateinputfromMFB
from TrainAndTest.common_func import create_model

# Version conflict

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor

    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
import warnings

warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Data options
parser.add_argument('--train-dir', type=str, help='path to dataset')
parser.add_argument('--test-dir', type=str, help='path to voxceleb1 test dataset')
parser.add_argument('--sitw-dir', type=str, help='path to voxceleb1 test dataset')

parser.add_argument('--test-only', action='store_true', default=False, help='using Cosine similarity')


parser.add_argument('--check-path', help='folder to output model checkpoints')
parser.add_argument('--extract-path', help='folder to output model grads, etc')

# Model options
# ALSTM  ASiResNet34  ExResNet34  LoResNet10  ResNet20  SiResNet34  SuResCNN10  TDNN
parser.add_argument('--model', type=str,
                    help='path to voxceleb1 test dataset')
parser.add_argument('--feat-dim', default=64, type=int, metavar='N',
                    help='acoustic feature dimension')

parser.add_argument('--revert', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--input-length', choices=['var', 'fix'], default='var',
                    help='choose the acoustic features type.')
parser.add_argument('--remove-vad', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--mvnorm', action='store_true', default=False,
                    help='using Cosine similarity')

parser.add_argument('--resnet-size', default=8, type=int,
                    metavar='RES', help='The channels of convs layers)')
parser.add_argument('--channels', default='64,128,256', type=str,
                    metavar='CHA', help='The channels of convs layers)')
parser.add_argument('--kernel-size', default='5,5', type=str, metavar='KE',
                    help='kernel size of conv filters')
parser.add_argument('--stride', default=2, type=int, metavar='ST',
                    help='kernel size of conv filters')
parser.add_argument('--time-dim', default=2, type=int, metavar='FEAT',
                    help='acoustic feature dimension')
parser.add_argument('--avg-size', type=int, default=4, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--start-epochs', type=int, default=36, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--epochs', type=int, default=36, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--loss-type', type=str, default='soft', choices=['soft', 'asoft', 'center', 'amsoft'],
                    help='path to voxceleb1 test dataset')
parser.add_argument('--dropout-p', type=float, default=0., metavar='BST',
                    help='input batch size for testing (default: 64)')

# args for additive margin-softmax
parser.add_argument('--margin', type=float, default=0.3, metavar='MARGIN',
                    help='the margin value for the angualr softmax loss function (default: 3.0')
parser.add_argument('--s', type=float, default=15, metavar='S',
                    help='the margin value for the angualr softmax loss function (default: 3.0')
# args for a-softmax
parser.add_argument('--m', type=int, default=3, metavar='M',
                    help='the margin value for the angualr softmax loss function (default: 3.0')
parser.add_argument('--lambda-min', type=int, default=5, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--lambda-max', type=float, default=0.05, metavar='S',
                    help='random seed (default: 0)')

parser.add_argument('--alpha', default=12, type=float, metavar='FEAT',
                    help='acoustic feature dimension')
parser.add_argument('--cos-sim', action='store_true', default=True, help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, metavar='ES', help='Dimensionality of the embedding')
parser.add_argument('--sample-utt', type=int, default=120, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--nj', default=12, type=int, metavar='NJOB', help='num of job')
parser.add_argument('--batch-size', type=int, default=1, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--input-per-spks', type=int, default=192, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=123456, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=1, metavar='LI',
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
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.multiprocessing.set_sharing_strategy('file_system')

if args.cuda:
    cudnn.benchmark = True

# Define visulaize SummaryWriter instance
kwargs = {'num_workers': args.nj, 'pin_memory': False} if args.cuda else {}
l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else PairwiseDistance(2)

if args.input_length == 'var':
    transform = transforms.Compose([
        # concateinputfromMFB(num_frames=c.NUM_FRAMES_SPECT, remove_vad=False),
        varLengthFeat(remove_vad=args.remove_vad),
        to2tensor()
    ])
    transform_T = transforms.Compose([
        # concateinputfromMFB(num_frames=c.NUM_FRAMES_SPECT, input_per_file=args.test_input_per_file, remove_vad=False),
        varLengthFeat(remove_vad=args.remove_vad),
        to2tensor()
    ])
elif args.input_length == 'fix':
    transform = transforms.Compose([
        concateinputfromMFB(remove_vad=args.remove_vad),
        to2tensor()
    ])
    transform_T = transforms.Compose([
        concateinputfromMFB(input_per_file=args.test_input_per_file, remove_vad=args.remove_vad),
        to2tensor()
    ])

if args.mvnorm:
    transform.transforms.append(mvnormal())
    transform_T.transforms.append(mvnormal())

file_loader = read_mat

train_dir = ScriptTrainDataset(dir=args.train_dir, samples_per_speaker=args.input_per_spks,
                               loader=file_loader, transform=transform, return_uid=True)
indices = list(range(len(train_dir)))
random.shuffle(indices)
indices = indices[:args.sample_utt]
train_part = torch.utils.data.Subset(train_dir, indices)

veri_dir = ScriptTestDataset(dir=args.train_dir, loader=file_loader, transform=transform_T, return_uid=True)
veri_dir.partition(args.sample_utt)

test_dir = ScriptTestDataset(dir=args.test_dir, loader=file_loader, transform=transform_T, return_uid=True)
test_dir.partition(args.sample_utt)

valid_dir = ScriptValidDataset(valid_set=train_dir.valid_set, spk_to_idx=train_dir.spk_to_idx,
                               valid_uid2feat=train_dir.valid_uid2feat, valid_utt2spk_dict=train_dir.valid_utt2spk_dict,
                               loader=file_loader, transform=transform, return_uid=True)
indices = list(range(len(valid_dir)))
random.shuffle(indices)
indices = indices[:args.sample_utt]
valid_part = torch.utils.data.Subset(valid_dir, indices)

# sitw_test_dir = SitwTestDataset(sitw_dir=args.sitw_dir, sitw_set='eval', transform=transform_T, return_uid=False)
# indices = list(range(len(sitw_test_dir)))
# random.shuffle(indices)
# indices = indices[:args.sample_utt]
# sitw_test_part = torch.utils.data.Subset(sitw_test_dir, indices)
#
# sitw_dev_dir = SitwTestDataset(sitw_dir=args.sitw_dir, sitw_set='dev', transform=transform_T, return_uid=False)
# indices = list(range(len(sitw_dev_dir)))
# random.shuffle(indices)
# indices = indices[:args.sample_utt]
# sitw_dev_part = torch.utils.data.Subset(sitw_dev_dir, indices)


def train_extract(train_loader, model, file_dir, set_name, save_per_num=2500):
    # switch to evaluate mode
    model.eval()

    input_grads = []
    inputs_uids = []
    pbar = tqdm(enumerate(train_loader))

    for batch_idx, (data, label, uid) in pbar:

        # orig = data.detach().numpy().squeeze().astype(np.float32)
        data = Variable(data.cuda(), requires_grad=True)

        logit, _ = model(data)

        if args.loss_type == 'asoft':
            classifed, _ = logit
        else:
            classifed = logit
        # conv1 = model.conv1(data)
        # bn1 = model.bn1(conv1)
        # relu1 = model.relu(bn1)
        # conv1 = conv1.cpu().detach().numpy().squeeze().astype(np.float32)
        # bn1 = bn1.cpu().detach().numpy().squeeze().astype(np.float32)
        # relu1 = relu1.cpu().detach().numpy().squeeze().astype(np.float32)

        classifed[0][label.long()].backward()

        grad = data.grad.cpu().numpy().squeeze().astype(np.float32)
        data = data.data.cpu().numpy().squeeze().astype(np.float32)
        if args.revert:
            grad = grad.transpose()
            data = data.transpose()

        input_grads.append([data, grad])
        inputs_uids.append(uid)

        model.zero_grad()

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Saving {} : [{:8d}/{:8d} ({:3.0f}%)] '.format(
                uid,
                batch_idx + 1,
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader)))

        if (batch_idx + 1) % save_per_num == 0 or (batch_idx + 1) == len(train_loader.dataset):
            num = batch_idx // save_per_num if batch_idx + 1 % save_per_num == 0 else batch_idx // save_per_num + 1
            # checkpoint_dir / extract / < dataset > / < set >.*.bin

            filename = file_dir + '/%s.%d.bin' % (set_name, num)
            with open(filename, 'wb') as f:
                pickle.dump(input_grads, f)

            with open(file_dir + '/inputs.%s.%d.json' % (set_name, num), 'w') as f:
                json.dump(inputs_uids, f)

            input_grads = []
            inputs_uids = []

    print('Saving pairs in %s.\n' % file_dir)
    torch.cuda.empty_cache()


def test_extract(test_loader, model, file_dir, set_name, save_per_num=1500):
    # switch to evaluate mode
    model.eval()

    input_grads = []
    inputs_uids = []
    pbar = tqdm(enumerate(test_loader))

    # for batch_idx, (data_a, data_b, label) in pbar:
    for batch_idx, (data_a, data_b, label, uid_a, uid_b) in pbar:

        data_a = Variable(data_a.cuda(), requires_grad=True)
        data_b = Variable(data_b.cuda(), requires_grad=True)

        _, feat_a = model(data_a)
        _, feat_b = model(data_b)

        cos_sim = l2_dist(feat_a, feat_b)
        cos_sim[0].backward()

        grad_a = data_a.grad.cpu().numpy().squeeze().astype(np.float32)
        grad_b = data_a.grad.cpu().numpy().squeeze().astype(np.float32)
        data_a = data_a.data.cpu().numpy().squeeze().astype(np.float32)
        data_b = data_b.data.cpu().numpy().squeeze().astype(np.float32)

        if args.revert:
            grad_a = grad_a.transpose()
            data_a = data_a.transpose()

            grad_b = grad_b.transpose()
            data_b = data_b.transpose()

        input_grads.append((label, grad_a, grad_b, data_a, data_b))
        inputs_uids.append([uid_a, uid_b])

        model.zero_grad()

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Saving pair [{:8d}/{:8d} ({:3.0f}%)] '.format(
                batch_idx + 1,
                len(test_loader),
                100. * batch_idx / len(test_loader)))

        if (batch_idx + 1) % save_per_num == 0 or (batch_idx + 1) == len(test_loader.dataset):
            num = batch_idx // save_per_num if batch_idx + 1 % save_per_num == 0 else batch_idx // save_per_num + 1
            # checkpoint_dir / extract / < dataset > / < set >.*.bin

            filename = file_dir + '/%s.%d.bin' % (set_name, num)
            # print('Saving pairs in %s.' % filename)

            with open(filename, 'wb') as f:
                pickle.dump(input_grads, f)

            with open(file_dir + '/inputs.%s.%d.json' % (set_name, num), 'w') as f:
                json.dump(inputs_uids, f)

            input_grads = []
            inputs_uids = []
    print('Saving pairs into %s.\n' % file_dir)
    torch.cuda.empty_cache()

def main():
    print('\nNumber of Speakers: {}.'.format(train_dir.num_spks))
    # print the experiment configuration
    print('Current time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))

    # instantiate model and initialize weights

    channels = args.channels.split(',')
    channels = [int(x) for x in channels]

    kernel_size = args.kernel_size.split(',')
    kernel_size = [int(x) for x in kernel_size]
    padding = [int((x - 1) / 2) for x in kernel_size]

    kernel_size = tuple(kernel_size)
    padding = tuple(padding)

    model_kwargs = {'input_dim': args.feat_dim,
                    'kernel_size': kernel_size,
                    'stride': args.stride,
                    'padding': padding,
                    'channels': channels,
                    'alpha': args.alpha,
                    'avg_size': args.avg_size,
                    'time_dim': args.time_dim,
                    'resnet_size': args.resnet_size,
                    'embedding_size': args.embedding_size,
                    'time_dim': args.time_dim,
                    'num_classes': len(train_dir.speakers),
                    'dropout_p': args.dropout_p}

    print('Model options: {}'.format(model_kwargs))

    model = create_model(args.model, **model_kwargs)
    if args.loss_type == 'asoft':
        model.classifier = AngleLinear(in_features=args.embedding_size, out_features=train_dir.num_spks, m=args.m)
    elif args.loss_type == 'amsoft':
        model.classifier = AdditiveMarginLinear(feat_dim=args.embedding_size, n_classes=train_dir.num_spks)

    train_loader = DataLoader(train_part, batch_size=args.batch_size, shuffle=False, **kwargs)
    veri_loader = DataLoader(veri_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    valid_loader = DataLoader(valid_part, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    # sitw_test_loader = DataLoader(sitw_test_part, batch_size=args.batch_size, shuffle=False, **kwargs)
    # sitw_dev_loader = DataLoader(sitw_dev_part, batch_size=args.batch_size, shuffle=False, **kwargs)

    resume_path = args.check_path + '/checkpoint_{}.pth'
    print('=> Saving output in {}\n'.format(args.extract_path))
    epochs = np.arange(args.start_epochs, args.epochs + 1)

    for e in epochs:
        # Load model from Checkpoint file
        if os.path.isfile(resume_path.format(e)):
            print('=> loading checkpoint {}'.format(resume_path.format(e)))
            checkpoint = torch.load(resume_path.format(e))
            # epoch = checkpoint['epoch']
            if e == 0:
                filtered = checkpoint.state_dict()
            else:
                filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}

            # model.load_state_dict(filtered)
            model_dict = model.state_dict()
            model_dict.update(filtered)
            model.load_state_dict(model_dict)

            try:
                args.dropout_p = model.dropout_p
            except:
                pass
        else:
            print('=> no checkpoint found at %s' % resume_path.format(e))
            continue
        model.cuda()

        file_dir = args.extract_path + '/epoch_%d' % e
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        if not args.test_only:
            # if args.cuda:
            #     model_conv1 = model.conv1.weight.cpu().detach().numpy()
            #     np.save(file_dir + '/model.conv1.npy', model_conv1)

            train_extract(train_loader, model, file_dir, 'vox1_train')
            train_extract(valid_loader, model, file_dir, 'vox1_valid')
            test_extract(veri_loader, model, file_dir, 'vox1_veri')

        test_extract(test_loader, model, file_dir, 'vox1_test')


if __name__ == '__main__':
    main()
