#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: train_surescnn10_kaldi.py
@Time: 2020/3/3 4:11 PM
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
parser.add_argument('--train-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev',
                    help='path to dataset')
parser.add_argument('--test-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test',
                    help='path to voxceleb1 test dataset')
parser.add_argument('--sitw-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/sitw_spect',
                    help='path to voxceleb1 test dataset')

parser.add_argument('--check-path', default='Data/checkpoint/SuResCNN10/spect/kaldi_final',
                    help='folder to output model checkpoints')
parser.add_argument('--resume',
                    default='Data/checkpoint/SuResCNN10/spect/kaldi_final/checkpoint_20.pth', type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=22, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--min-softmax-epoch', type=int, default=40, metavar='MINEPOCH',
                    help='minimum epoch for initial parameter using softmax (default: 2')

# Training options
parser.add_argument('--feat-dim', default=161, type=int, metavar='N',
                    help='acoustic feature dimension')
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--input-per-spks', type=int, default=192, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-input-per-file', type=int, default=4, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-batch-size', type=int, default=4, metavar='BST',
                    help='input batch size for testing (default: 64)')

# parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
parser.add_argument('--margin', type=float, default=3, metavar='MARGIN',
                    help='the margin value for the angualr softmax loss function (default: 3.0')
parser.add_argument('--loss-ratio', type=float, default=2.0, metavar='LOSSRATIO',
                    help='the ratio softmax loss - triplet loss (default: 2.0')

# args for a-softmax
parser.add_argument('--lambda-min', type=int, default=5, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--lambda-max', type=int, default=1000, metavar='S',
                    help='random seed (default: 0)')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=0, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='W', help='momentum for sgd (default: 0.9)')
parser.add_argument('--dampening', default=0, type=float,
                    metavar='W', help='dampening for sgd (default: 0.0)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')

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

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.multiprocessing.set_sharing_strategy('file_system')

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

# create logger
# Define visulaize SummaryWriter instance
writer = SummaryWriter(logdir=args.check_path, filename_suffix='_final')

kwargs = {'num_workers': 12, 'pin_memory': True} if args.cuda else {}
if not os.path.exists(args.check_path):
    os.makedirs(args.check_path)

opt_kwargs = {'lr': args.lr,
              'lr_decay': args.lr_decay,
              'weight_decay': args.weight_decay,
              'dampening': args.dampening,
              'momentum': args.momentum}

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

else:
    transform = transforms.Compose([
        truncatedinput(),
        toMFB(),
        totensor(),
        # tonormal()
    ])
    file_loader = read_audio

# pdb.set_trace()
file_loader = read_mat
train_dir = ScriptTrainDataset(dir=args.train_dir, samples_per_speaker=args.input_per_spks, loader=file_loader,
                               transform=transform)
test_dir = ScriptTestDataset(dir=args.test_dir, loader=file_loader, transform=transform_T)

indices = list(range(len(test_dir)))
random.shuffle(indices)
indices = indices[:12800]
test_part = torch.utils.data.Subset(test_dir, indices)

# sitw_test_dir = SitwTestDataset(sitw_dir=args.sitw_dir, sitw_set='eval', transform=transform_T)
# indices = list(range(len(sitw_test_dir)))
# random.shuffle(indices)
# indices = indices[:12800]
# sitw_test_part = torch.utils.data.Subset(sitw_test_dir, indices)
#
# sitw_dev_dir = SitwTestDataset(sitw_dir=args.sitw_dir, sitw_set='dev', transform=transform_T)
# indices = list(range(len(sitw_dev_dir)))
# random.shuffle(indices)
# indices = indices[:12800]
# sitw_dev_part = torch.utils.data.Subset(sitw_dev_dir, indices)

valid_dir = ScriptValidDataset(valid_set=train_dir.valid_set, loader=file_loader, spk_to_idx=train_dir.spk_to_idx,
                               valid_uid2feat=train_dir.valid_uid2feat, valid_utt2spk_dict=train_dir.valid_utt2spk_dict,
                               transform=transform)

def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # test_display_triplet_distance = False
    # print the experiment configuration
    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))
    print('Number of Speakers: {}.\n'.format(train_dir.num_spks))

    # instantiate model and initialize weights
    model = SuperficialResCNN(layers=[1, 1, 1, 0], embedding_size=args.embedding_size,
                              n_classes=train_dir.num_spks, m=args.margin)

    optimizer = create_optimizer(model.parameters(), args.optimizer, **opt_kwargs)
    scheduler = MultiStepLR(optimizer, milestones=[9, 14, 19], gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
            model.load_state_dict(filtered)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            # criterion.load_state_dict(checkpoint['criterion'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    print('start epoch is : ' + str(start) + '\n')
    # start = 0
    end = start + args.epochs

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=True,
                                               **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(args.batch_size / 2), shuffle=False,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(test_part, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    # sitw_test_loader = torch.utils.data.DataLoader(sitw_test_part, batch_size=args.test_batch_size, shuffle=False,
    #                                                **kwargs)
    # sitw_dev_loader = torch.utils.data.DataLoader(sitw_dev_part, batch_size=args.test_batch_size, shuffle=False,
    #                                               **kwargs)

    ce = AngleSoftmaxLoss(lambda_min=args.lambda_min, lambda_max=args.lambda_max).cuda()
    # ce = nn.CrossEntropyLoss().cuda()

    check_path = '{}/checkpoint_{}.pth'.format(args.check_path, 0)
    torch.save({'epoch': 0, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
               # 'criterion': criterion.state_dict()
               check_path)

    if args.cuda:
        model = model.cuda()
        ce = ce.cuda()

    for epoch in range(start, end):
        # pdb.set_trace()
        for param_group in optimizer.param_groups:
            print('\n\33[1;34m Current \'{}\' learning rate is {}.\33[0m'.format(args.optimizer, param_group['lr']))

        train(train_loader, model, ce, optimizer, scheduler, epoch)
        test(test_loader, valid_loader, model, epoch)
        # sitw_test(sitw_test_loader, model, epoch)
        # sitw_test(sitw_dev_loader, model, epoch)

        scheduler.step()
        # exit(1)

    writer.close()


def train(train_loader, model, ce, optimizer, scheduler, epoch):
    # switch to evaluate mode
    model.train()

    correct = 0.
    total_datasize = 0.
    total_loss = 0.
    # for param_group in optimizer.param_groups:
    #     print('\33[1;34m Optimizer \'{}\' learning rate is {}.\33[0m'.format(args.optimizer, param_group['lr']))

    pbar = tqdm(enumerate(train_loader))
    output_softmax = nn.Softmax(dim=1)

    for batch_idx, (data, label) in pbar:
        if args.cuda:
            data = data.float().cuda()
        data, label = Variable(data), Variable(label)

        # pdb.set_trace()
        classfier, _ = model(data)
        true_labels = label.cuda()
        cos_theta, phi_theta = classfier

        loss = ce(classfier, true_labels)

        predicted_labels = output_softmax(cos_theta)
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]
        minibatch_acc = float((predicted_one_labels.cuda() == true_labels.cuda()).sum().item()) / len(
            predicted_one_labels)
        correct += float((predicted_one_labels.cuda() == true_labels.cuda()).sum().item())
        total_datasize += len(predicted_one_labels)
        total_loss += loss.item()

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch {:2d}: [{:8d}/{:8d} ({:3.0f}%)] Avg Loss: {:.4f} Batch Accuracy: {:.4f}%'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    total_loss / (batch_idx + 1),
                    100. * minibatch_acc))

    check_path = '{}/checkpoint_{}.pth'.format(args.check_path, epoch)
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
               # 'criterion': criterion.state_dict()
               check_path)

    print('\n\33[91mTrain Epoch {}: Train Accuracy:{:.6f}%, Avg loss: {}.\n\33[0m'.format(epoch, 100 * float(
        correct) / total_datasize, total_loss / len(train_loader)))
    writer.add_scalar('Train/Accuracy', correct / total_datasize, epoch)
    writer.add_scalar('Train/Loss', total_loss / len(train_loader), epoch)

    torch.cuda.empty_cache()

def test(test_loader, valid_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    valid_pbar = tqdm(enumerate(valid_loader))
    softmax = nn.Softmax(dim=1)

    correct = 0.
    total_datasize = 0.

    for batch_idx, (data, label) in valid_pbar:
        data = Variable(data.cuda())

        # compute output
        out, _ = model(data)
        cos_theta, phi_theta = out
        predicted_labels = cos_theta
        true_labels = Variable(label.cuda())

        # pdb.set_trace()
        predicted_one_labels = softmax(predicted_labels)
        predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]

        batch_correct = (predicted_one_labels.cuda() == true_labels.cuda()).sum().item()
        minibatch_acc = float(batch_correct / len(predicted_one_labels))
        correct += batch_correct
        total_datasize += len(predicted_one_labels)

        if batch_idx % args.log_interval == 0:
            valid_pbar.set_description('Valid Epoch: {:2d} [{:8d}/{:8d} ({:3.0f}%)] Batch Accuracy: {:.4f}%'.format(
                    epoch,
                    batch_idx * len(data),
                    len(valid_loader.dataset),
                    100. * batch_idx / len(valid_loader),
                    100. * minibatch_acc
                ))

    valid_accuracy = 100. * correct / total_datasize
    writer.add_scalar('Test/Valid_Accuracy', valid_accuracy, epoch)
    torch.cuda.empty_cache()

    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
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
                epoch, batch_idx * len(data_a), len(test_loader.dataset), 100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    writer.add_scalar('Test/EER', 100. * eer, epoch)
    writer.add_scalar('Test/Threshold', eer_threshold, epoch)

    print('\n\33[91mFor {}_distance, Test ERR is {:.4f}%, Threshold is {}. Valid ' \
          'Accuracy is {:.2f}%.\33[0m \n'.format('cos' if args.cos_sim else 'l2', 100. * eer,
                                                                  eer_threshold, valid_accuracy))
    torch.cuda.empty_cache()

def sitw_test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
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
                epoch, batch_idx * len(data_a) / vec_shape[1], len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    writer.add_scalar('Test/SITW_EER', 100. * eer, epoch)
    writer.add_scalar('Test/SITW_Threshold', eer_threshold, epoch)

    print('\33[91mFor Sitw Test ERR is {:.4f}%, Threshold is {}.\n\33[0m'.format(100. * eer, eer_threshold))

# python TrainAndTest/Spectrogram/train_surescnn10_kaldi.py > Log/SuResCNN10/spect_161/

if __name__ == '__main__':
    main()
