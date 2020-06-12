#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: train_exres_kaldi.py
@Time: 2020/3/27 10:46 AM
@Overview:
"""
# from __future__ import print_function
import argparse
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from kaldi_io import read_mat, read_vec_flt
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from Define_Model.LossFunction import CenterLoss
from Define_Model.SoftmaxLoss import AngleSoftmaxLoss, AngleLinear, AdditiveMarginLinear, AMSoftmaxLoss
from Define_Model.model import PairwiseDistance
from Process_Data.KaldiDataset import ScriptTrainDataset, ScriptTestDataset, ScriptValidDataset, KaldiExtractDataset, \
    ScriptVerifyDataset
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, concateinputfromMFB, to2tensor, mvnormal
from TrainAndTest.common_func import create_optimizer, create_model, verification_extract, verification_test
from eval_metrics import evaluate_kaldi_eer, evaluate_kaldi_mindcf
from logger import NewLogger

warnings.filterwarnings("ignore")
# Version conflict

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
# options for vox1
parser.add_argument('--train-dir', type=str,
                    help='path to dataset')
parser.add_argument('--test-dir', type=str,
                    help='path to voxceleb1 test dataset')
parser.add_argument('--nj', default=14, type=int, metavar='NJOB', help='num of job')

# Model options
parser.add_argument('--model', type=str,
                    help='path to voxceleb1 test dataset')
parser.add_argument('--resnet-size', default=34, type=int,
                    metavar='RES', help='The channels of convs layers)')
parser.add_argument('--kernel-size', default='5,5', type=str, metavar='KE',
                    help='kernel size of conv filters')
parser.add_argument('--stride', default=2, type=int, metavar='ST',
                    help='kernel size of conv filters')
parser.add_argument('--feat-dim', default=64, type=int, metavar='N',
                    help='acoustic feature dimension')
parser.add_argument('--dropout-p', type=float, default=0., metavar='BST',
                    help='input batch size for testing (default: 64)')

parser.add_argument('--avg-size', type=int, default=4, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--time-dim', default=2, type=int, metavar='FEAT',
                    help='acoustic feature dimension')

parser.add_argument('--check-path',
                    help='folder to output model checkpoints')
parser.add_argument('--save-init', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--resume',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=30, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--milestones', default='14,20,25', type=str,
                    metavar='MIL', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--veri-pairs', type=int, default=12800, metavar='VP',
                    help='number of epochs to train (default: 10)')

# Training options
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--remove-vad', action='store_true', default=False,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=128, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--num-valid', type=int, default=5, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=4, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--input-per-spks', type=int, default=224, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

# loss configure
parser.add_argument('--loss-type', type=str, default='soft', choices=['soft', 'asoft', 'center', 'amsoft'],
                    help='path to voxceleb1 test dataset')
parser.add_argument('--m', type=float, default=4, metavar='M',
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
parser.add_argument('--log-interval', type=int, default=15, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument('--acoustic-feature', choices=['fbank', 'spectrogram', 'mfcc'], default='fbank',
                    help='choose the acoustic features type.')
parser.add_argument('--mfb', action='store_true', default=True,
                    help='start from MFB file')
parser.add_argument('--makemfb', action='store_true', default=False,
                    help='need to make mfb file')

args = parser.parse_args()

# Set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.multiprocessing.set_sharing_strategy('file_system')

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

# create logger
# Define visulaize SummaryWriter instance
writer = SummaryWriter(args.check_path, filename_suffix='test')
sys.stdout = NewLogger(os.path.join(args.check_path, 'log.txt'))

kwargs = {'num_workers': args.nj, 'pin_memory': True} if args.cuda else {}
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
        concateinputfromMFB(remove_vad=args.remove_vad),  # num_frames=np.random.randint(low=300, high=500)),
        to2tensor(),
        mvnormal()
    ])
    transform_T = transforms.Compose([
        concateinputfromMFB(input_per_file=args.test_input_per_file, remove_vad=args.remove_vad),
        to2tensor(),
        mvnormal()
    ])
    file_loader = read_mat

else:
    transform = transforms.Compose([
        truncatedinput(),
        toMFB(),
        totensor(),
        # tonormal()
    ])

train_dir = ScriptTrainDataset(dir=args.train_dir, samples_per_speaker=args.input_per_spks, transform=transform,
                               loader=file_loader, num_valid=args.num_valid)
test_dir = ScriptTestDataset(dir=args.test_dir, transform=transform_T, loader=file_loader)
if len(test_dir) < args.veri_pairs:
    args.veri_pairs = len(test_dir)
    print('There are %d verification pairs.' % len(test_dir))
else:
    test_dir.partition(args.veri_pairs)

valid_dir = ScriptValidDataset(valid_set=train_dir.valid_set, spk_to_idx=train_dir.spk_to_idx,
                               valid_uid2feat=train_dir.valid_uid2feat,
                               valid_utt2spk_dict=train_dir.valid_utt2spk_dict,
                               transform=transform, loader=file_loader)

def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # print the experiment configuration
    print('\nCurrent time is \33[91m{}\33[0m'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))
    print('Number of Classes: {}\n'.format(train_dir.num_spks))

    # instantiate
    # model and initialize weights
    # instantiate model and initialize weights
    kernel_size = args.kernel_size.split(',')
    kernel_size = [int(x) for x in kernel_size]
    padding = [int((x - 1) / 2) for x in kernel_size]

    kernel_size = tuple(kernel_size)
    padding = tuple(padding)

    model_kwargs = {'input_dim': args.feat_dim,
                    'kernel_size': kernel_size,
                    'stride': args.stride,
                    'avg_size': args.avg_size,
                    'time_dim': args.time_dim,
                    'padding': padding,
                    'resnet_size': args.resnet_size,
                    'embedding_size': args.embedding_size,
                    'num_classes': len(train_dir.speakers),
                    'dropout_p': args.dropout_p}

    print('Model options: {}'.format(model_kwargs))

    model = create_model(args.model, **model_kwargs)

    start = 1
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
            model.load_state_dict(filtered)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            # criterion.load_state_dict(checkpoint['criterion'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    ce_criterion = nn.CrossEntropyLoss()
    if args.loss_type == 'soft':
        xe_criterion = None
    elif args.loss_type == 'asoft':
        ce_criterion = None
        model.classifier = AngleLinear(in_features=args.embedding_size, out_features=train_dir.num_spks, m=args.m)
        xe_criterion = AngleSoftmaxLoss(lambda_min=args.lambda_min, lambda_max=args.lambda_max)
    elif args.loss_type == 'center':
        xe_criterion = CenterLoss(num_classes=train_dir.num_spks, feat_dim=args.embedding_size)
    elif args.loss_type == 'amsoft':
        model.classifier = AdditiveMarginLinear(feat_dim=args.embedding_size, n_classes=train_dir.num_spks)
        xe_criterion = AMSoftmaxLoss(margin=args.margin, s=args.s)

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model.parameters(), args.optimizer, **opt_kwargs)
    if args.loss_type == 'center':
        optimizer = torch.optim.SGD([{'params': xe_criterion.parameters(), 'lr': args.lr * 5},
                                     {'params': model.parameters()}],
                                    lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum)

    milestones = args.milestones.split(',')
    milestones = [int(x) for x in milestones]
    milestones.sort()
    # print('Scheduler options: {}'.format(milestones))
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    if args.save_init and not args.finetune:
        check_path = '{}/checkpoint_{}.pth'.format(args.check_path, start)
        torch.save({'epoch': start, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}, check_path)

    start += args.start_epoch
    print('Start epoch is : ' + str(start))
    end = args.epochs + 1

    # pdb.set_trace()
    train_loader = torch.utils.data.DataLoader(train_dir,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dir,
                                               batch_size=int(args.batch_size / 2),
                                               shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dir,
                                              batch_size=args.test_batch_size,
                                              shuffle=False, **kwargs)

    ce = [ce_criterion, xe_criterion]
    if args.cuda:
        model = model.cuda()
        for i in range(len(ce)):
            if ce[i] != None:
                ce[i] = ce[i].cuda()

    for epoch in range(start, end):
        # pdb.set_trace()
        print('\n\33[1;34m Current \'{}\' learning rate is '.format(args.optimizer), end='')
        for param_group in optimizer.param_groups:
            print('{:.5f} '.format(param_group['lr']), end='')
        print(' \33[0m')

        train(train_loader, model, optimizer, ce, scheduler, epoch)
        test(test_loader, valid_loader, model, epoch)

        scheduler.step()
        # break
    verfify_dir = KaldiExtractDataset(dir=args.test_dir, transform=transform_T, filer_loader=file_loader)
    verify_loader = torch.utils.data.DataLoader(verfify_dir, batch_size=args.test_batch_size, shuffle=False,
                                                **kwargs)
    verification_extract(verify_loader, model, args.xvector_dir)
    file_loader = read_vec_flt
    test_dir = ScriptVerifyDataset(dir=args.test_dir, trials_file=args.trials,
                                   xvectors_dir=args.xvector_dir, loader=file_loader)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size * 64, shuffle=False, **kwargs)
    verification_test(test_loader=test_loader, dist_type='cos' if args.cos_sim else 'l2',
                      log_interval=args.log_interval)

    writer.close()


def train(train_loader, model, optimizer, ce, scheduler, epoch):
    # switch to evaluate mode
    model.train()

    correct = 0.
    total_datasize = 0.
    total_loss = 0.
    output_softmax = nn.Softmax(dim=1)
    ce_criterion, xe_criterion = ce

    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data, label) in pbar:

        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        data, true_labels = Variable(data), Variable(label)

        # pdb.set_trace()
        classfier, feats = model(data)
        classfier_label = classfier

        if args.loss_type == 'soft':
            loss = ce_criterion(classfier, true_labels)
        elif args.loss_type == 'asoft':
            classfier_label, _ = classfier
            loss = xe_criterion(classfier, true_labels)
        elif args.loss_type == 'center':
            loss_cent = ce_criterion(classfier, true_labels)
            loss_xent = xe_criterion(feats, true_labels)
            loss = args.loss_ratio * loss_xent + loss_cent
        elif args.loss_type == 'amsoft':
            loss = xe_criterion(classfier, true_labels)

        predicted_labels = output_softmax(classfier_label)
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]
        minibatch_correct = float((predicted_one_labels.cuda() == true_labels.cuda()).sum().item())
        minibatch_acc = minibatch_correct / len(predicted_one_labels)
        correct += minibatch_correct

        total_datasize += len(predicted_one_labels)
        total_loss += float(loss.item())
        # pdb.set_trace()
        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()

        if args.loss_type == 'center' and args.loss_ratio != 0:
            for param in xe_criterion.parameters():
                param.grad.data *= (1. / args.loss_ratio)

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

    # options for vox1
    if epoch % 2 == 1 or epoch == args.epochs:
        check_path = '{}/checkpoint_{}.pth'.format(args.check_path, epoch)
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'criterion': ce},
                   check_path)

    print('For {} with {} Epoch {:2d}: \n\33[91m Train set Accuracy:{:.4f}%.' \
          ' Average loss is {:.4f}.\33[0m\n'.format(args.model,
                                                    args.loss_type,
                                                    epoch,
                                                    100 * float(correct) / total_datasize,
                                                    total_loss / len(train_loader)))

    writer.add_scalar('Train/Accuracy', 100. * correct / total_datasize, epoch)
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
        # print(model.conv1.weight)
        # print(data)
        # pdb.set_trace()
        # compute output
        out, _ = model(data)
        if args.loss_type == 'asoft':
            predicted_labels, _ = out
        else:
            predicted_labels = out

        true_labels = Variable(label.cuda())

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
    # writer.add_scalar('Test/Valid_Accuracy', valid_accuracy, epoch)

    torch.cuda.empty_cache()

    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        vec_shape = data_a.shape
        # pdb.set_trace()
        if vec_shape[1] != 1:
            data_a = data_a.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])
            data_p = data_p.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a), Variable(data_p), Variable(label)

        # compute output
        _, out_a = model(data_a)
        _, out_p = model(data_p)


        if vec_shape[1] != 1:
            # dists = dists.reshape(vec_shape[0], vec_shape[1]).mean(axis=1)
            out_a = out_a.reshape(vec_shape[0], vec_shape[1], args.embedding_size).mean(axis=1)
            out_p = out_p.reshape(vec_shape[0], vec_shape[1], args.embedding_size).mean(axis=1)

        dists = l2_dist.forward(out_a, out_p)
        dists = dists.data.cpu().numpy()
        distances.append(dists)
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a) / args.test_input_per_file, len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    # err, accuracy= evaluate_eer(distances,labels)
    eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    writer.add_scalar('Test/EER', 100. * eer, epoch)
    writer.add_scalar('Test/Threshold', eer_threshold, epoch)

    mindcf_01, mindcf_001 = evaluate_kaldi_mindcf(distances, labels)
    writer.add_scalar('Test/mindcf-0.01', mindcf_01, epoch)
    writer.add_scalar('Test/mindcf-0.001', mindcf_001, epoch)

    dist_type = 'cos' if args.cos_sim else 'l2'
    print('For %s_distance, ' % dist_type)
    print('  \33[91mTest ERR is {:.4f}%, Threshold is {}'.format(100. * eer, eer_threshold))
    print('  mindcf-0.01 {:.4f}, mindcf-0.001 {:.4f},'.format(mindcf_01, mindcf_001))
    print('  Valid Accuracy is %.4f %%.\33[0m\n' % valid_accuracy)


    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
