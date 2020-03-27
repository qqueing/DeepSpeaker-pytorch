#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: train_tdnn.py
@Time: 2019/9/19 上午11:42
@Overview: Todo: extract fbank24 for vox1 and train TDNN
"""
from __future__ import print_function

import random
import time

import torch.nn as nn
import torch
import pathlib
import os
import numpy as np
from tensorboardX import SummaryWriter
from torch import optim

from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
# import trainset
# import testset
from tqdm import tqdm

from Define_Model.TDNN import Time_Delay
from torch.utils.data import DataLoader
import warnings
import argparse
import pdb

from Process_Data.DeepSpeakerDataset_dynamic import ClassificationDataset, SpeakerTrainDataset, ValidationDataset
from Process_Data.VoxcelebTestset import VoxcelebTestset
from Process_Data.audio_processing import truncatedinputfromMFB, totensor, read_MFB, make_Fbank, conver_to_wav, \
    concateinputfromMFB, varLengthFeat, PadCollate
from Process_Data.voxceleb2_wav_reader import voxceleb2_list_reader
from Process_Data import constants as c
from Process_Data.voxceleb_wav_reader import wav_list_reader, wav_duration_reader, dic_dataset
from TrainAndTest.common_func import create_optimizer
from eval_metrics import evaluate_kaldi_eer

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='TDNN-based x-vector Speaker Recognition')

parser.add_argument('--dataroot', type=str, default='Data/dataset/voxceleb1/fbank24',
                    help='path to local dataset')
parser.add_argument('--test-dataroot', type=str, default='Data/dataset/voxceleb1/fbank24',
                    help='path to local dataset')
# parser.add_argument('--dataset', type=str, default='/home/cca01/work2019/Data/voxceleb2',
#                     help='path to dataset')
parser.add_argument('--test-pairs-path', type=str, default='Data/dataset/voxceleb1/test_trials/ver_list.txt',
                    help='path to pairs file')
parser.add_argument('--check-path', type=str, default='Data/checkpoint/TDNN/vox1/soft',
                    help='path to dataset')
parser.add_argument('--resume', default='Data/checkpoint/TDNN/vox1/soft/checkpoint_1.pth',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--acoustic-feature', type=str, default='fbank',
                    help='path to dataset')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=32, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--input-per-spks', type=int, default=200, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-input-per-file', type=int, default=2, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--make-feats', action='store_true', default=False,
                    help='need to make spectrograms file')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=0, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='W', help='momentum for sgd (default: 0.9)')
parser.add_argument('--dampening', default=0, type=float,
                    metavar='W', help='dampening for sgd (default: 0.0)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')

parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument('--gpu-id', default='2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# torch.cuda.set_device(int(args.gpu_id))

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    cudnn.benchmark = True

writer = SummaryWriter(logdir=args.check_path, filename_suffix='fb24_vox1')
# voxceleb, train_set, valid_set = wav_list_reader(args.dataroot, split=True)
voxceleb, train_set, valid_set = wav_duration_reader(data_path=args.dataroot)
train_dataset = dic_dataset(train_set)

# voxceleb, voxceleb_dev = wav_list_reader(args.dataset)
# vox2
# voxceleb, voxceleb_dev = voxceleb2_list_reader(args.dataset)

# pdb.set_trace()
if args.acoustic_feature == 'fbank':
    transform = transforms.Compose([
        concateinputfromMFB(),
        totensor()
    ])
    transform_T = transforms.Compose([
        concateinputfromMFB(input_per_file=args.test_input_per_file),
        # varLengthFeat(),
        totensor()
    ])
    file_loader = read_MFB

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
opt_kwargs = {'lr': args.lr,
              'lr_decay': args.lr_decay,
              'weight_decay': args.weight_decay,
              'dampening': args.dampening,
              'momentum': args.momentum}

l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else nn.PairwiseDistance(2)

# train_dir = ClassificationDataset(voxceleb=train_set, dir=args.dataroot, loader=file_loader, transform=transform)
train_dir = SpeakerTrainDataset(dataset=train_dataset, dir=args.dataroot, loader=file_loader, transform=transform,
                                feat_dim=24, samples_per_speaker=args.input_per_spks)
test_dir = VoxcelebTestset(dir=args.dataroot, pairs_path=args.test_pairs_path, loader=file_loader,
                           transform=transform_T)

indices = list(range(len(test_dir)))
random.shuffle(indices)
indices = indices[:4000]
test_part = torch.utils.data.Subset(test_dir, indices)

valid_dir = ValidationDataset(voxceleb=valid_set, dir=args.dataroot, loader=file_loader,
                              class_to_idx=train_dir.class_to_idx, transform=transform)

del voxceleb
del train_set
del valid_set


def main():
    # print the experiment configuration
    print('\33[91m \nCurrent time is {}\33[0m'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))
    print('Number of Speakers: {}\n'.format(len(train_dir.classes)))
    # device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

    context = [[-2, 2], [-2, 0, 2], [-3, 0, 3], [0], [0]]
    # the same configure as x-vector
    node_num = [512, 512, 512, 512, 1500, 3000, 512, 512]
    full_context = [True, False, False, True, True]

    # train_set = trainset.TrainSet('../all_feature/')
    # todo:
    # train_set = []
    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=PadCollate(dim=2), **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=args.batch_size, shuffle=False,
                                               collate_fn=PadCollate(dim=2), **kwargs)
    test_loader = torch.utils.data.DataLoader(test_part, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    # train_loader = DataLoader(train_dir, batch_size=args.batch_size, shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=args.batch_size, shuffle=False)
    # test_loader = DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False)

    model = Time_Delay(context, 24, len(train_dir.classes), node_num, full_context)

    if args.cuda:
        # model.to(device)
        model = model.cuda()

    optimizer = create_optimizer(model.parameters(), args.optimizer, **opt_kwargs)
    scheduler = MultiStepLR(optimizer, milestones=[16, 24], gamma=0.1)
    ce_loss = nn.CrossEntropyLoss().cuda()
    # torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # torch.set_num_threads(16)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
            model.load_state_dict(filtered)
            optimizer.load_state_dict(checkpoint['optimizer'])
            # criterion.load_state_dict(checkpoint['criterion'])

        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    print('start epoch is : ' + str(start))
    # start = 0
    end = start + args.epochs

    for epoch in range(start, end):
        # pdb.set_trace()
        train(train_loader, model, ce_loss, optimizer, epoch)
        test(test_loader, valid_loader, model, epoch)
        scheduler.step()


def train(train_loader, model, ce_loss, optimizer, epoch):
    # switch to evaluate mode
    model.train()

    running_loss = 0.
    total = 0
    correct = 0.

    output_softmax = nn.Softmax(dim=1)

    # learning rate multiple 0.1 per 15 epochs
    for param_group in optimizer.param_groups:
        print('\33[1;34m Current \'{}\' learning rate is {}.\33[0m'.format(args.optimizer, param_group['lr']))

    pbar = tqdm(enumerate(train_loader))
    # pdb.set_trace()

    for batch_idx, (data, label) in pbar:
        # break
        if args.cuda:
            data = data.cuda()
            label = label.cuda()

        data, label = Variable(data), Variable(label)

        xvectors = model.pre_forward(data)
        output = model(xvectors)

        loss = ce_loss(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted_one_labels = torch.max(output_softmax(output), dim=1)[1]
        batch_correct = float((predicted_one_labels.cuda() == label.cuda()).sum().item())
        # _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += batch_correct

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.4f}\tBatch Accuracy: {:.4f}%'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                    100. * float(batch_correct) / label.size(0)))

    check_path = pathlib.Path('{}/checkpoint_{}.pth'.format(args.check_path, epoch))
    if not check_path.parent.exists():
        os.makedirs(str(check_path.parent))

    torch.save({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               # 'criterion': criterion.state_dict()
               str(check_path))
    try:
        print('\33[1;34m Train epoch {}, Accuracy is {}%, and Avg loss: {:.4f}.\33[0m \n'.format(epoch, 100. * float(
            correct) / total, running_loss / len(train_loader)))
        writer.add_scalar('Train/Accuracy', 100. * float(correct) / total, epoch)
        writer.add_scalar('Train/Loss', running_loss / len(train_loader), epoch)
    except Exception:
        print('\033[1;34m Something wrong with logging!\033\n[0m')


def test(test_loader, valid_loader, model, epoch):
    # net = model.to('cuda:1')
    model.eval()

    valid_pbar = tqdm(enumerate(valid_loader))
    softmax = nn.Softmax(dim=1)

    correct = 0.
    total_datasize = 0.

    for batch_idx, (data, label) in valid_pbar:
        data = Variable(data.cuda())

        # compute output
        out = model.pre_forward(data)
        cls = model(out)

        predicted_labels = cls
        true_labels = Variable(label.cuda())

        # pdb.set_trace()
        predicted_one_labels = softmax(predicted_labels)
        predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]

        batch_correct = (predicted_one_labels.cuda() == true_labels.cuda()).sum().item()
        minibatch_acc = float(batch_correct / len(predicted_one_labels))
        correct += batch_correct
        total_datasize += len(predicted_one_labels)

        if batch_idx % args.log_interval == 0:
            valid_pbar.set_description(
                'Valid Epoch for Classification: {:2d} [{:8d}/{:8d} ({:3.0f}%)] Batch Accuracy: {:.4f}%'.format(
                    epoch,
                    batch_idx * len(data),
                    len(valid_loader.dataset),
                    100. * batch_idx / len(valid_loader),
                    100. * minibatch_acc
                ))

    valid_accuracy = 100. * correct / total_datasize
    writer.add_scalar('Test/Valid_Accuracy', valid_accuracy, epoch)
    # test_set = testset.TestSet('../all_feature/')
    # todo:
    # test_set = []
    distances = []
    labels = []
    pbar = tqdm(enumerate(test_loader))
    # pdb.set_trace()

    for batch_idx, (data_a, data_p, label) in pbar:
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        # compute output
        current_sample = data_a.size(0)
        data_a = data_a.resize_(args.test_input_per_file * current_sample, 1, data_a.size(2), data_a.size(3))
        data_p = data_p.resize_(args.test_input_per_file * current_sample, 1, data_a.size(2), data_a.size(3))

        x_vector_a, x_vector_p = model.pre_forward(data_a), model.pre_forward(data_p)

        dists = l2_dist.forward(x_vector_a, x_vector_p)
        dists = dists.data.cpu().numpy()
        dists = dists.reshape(current_sample, args.test_input_per_file).mean(axis=1)

        distances.append(dists)
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a) / args.test_input_per_file, len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    # err, accuracy= evaluate_eer(distances,labels)
    # err, accuracy= evaluate_eer(distances,labels)
    eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    writer.add_scalar('Test/EER', eer, epoch)
    writer.add_scalar('Test/Threshold', eer_threshold, epoch)
    # tpr, fpr, accuracy, val, far = evaluate(distances, labels)

    print('\33[91mFor {}_distance, Test set verification ERR is {:.4f}%, when threshold is {}. Valid ' \
          'set classificaton accuracy is {:.2f}%.\n\33[0m'.format('cos' if args.cos_sim else 'l2', 100. * eer,
                                                                  eer_threshold, valid_accuracy))


if __name__ == '__main__':
    main()
