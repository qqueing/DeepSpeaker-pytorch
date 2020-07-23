#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: train_lores10_kaldi.py
@Time: 2020/4/4 11:14 AM
@Overview:
"""
from __future__ import print_function

import argparse
import os
import os.path as osp
import sys
import time
# Version conflict
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from kaldi_io import read_mat, read_vec_flt
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from tqdm import tqdm

from Define_Model.LossFunction import CenterLoss
from Define_Model.SoftmaxLoss import AngleSoftmaxLoss, AngleLinear, AdditiveMarginLinear, AMSoftmaxLoss
from Define_Model.model import PairwiseDistance
from Process_Data import constants as c
from Process_Data.KaldiDataset import ScriptTrainDataset, ScriptTestDataset, ScriptValidDataset, KaldiExtractDataset, \
    ScriptVerifyDataset
from Process_Data.audio_processing import concateinputfromMFB, to2tensor, varLengthFeat
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, read_audio
from TrainAndTest.common_func import create_optimizer, create_model, verification_test, verification_extract
from eval_metrics import evaluate_kaldi_eer, evaluate_kaldi_mindcf
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
# Data options
parser.add_argument('--train-dir', type=str, help='path to dataset')
parser.add_argument('--test-dir', type=str, help='path to voxceleb1 test dataset')
parser.add_argument('--trials', type=str, default='trials', help='trials filename')
parser.add_argument('--domain', action='store_true', default=False, help='set domain in dataset')

parser.add_argument('--nj', default=12, type=int, metavar='NJOB', help='num of job')
parser.add_argument('--feat-format', type=str,
                    default='kaldi', choices=['kaldi', 'npy'],
                    help='number of jobs to make feats (default: 10)')

parser.add_argument('--check-path', default='Data/checkpoint/LoResNet10/spect/soft',
                    help='folder to output model checkpoints')
parser.add_argument('--save-init', action='store_true', default=True, help='need to make mfb file')
parser.add_argument('--resume',
                    default='Data/checkpoint/LoResNet10/spect/soft/checkpoint_10.pth', type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=20, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--scheduler', default='multi', type=str,
                    metavar='SCH', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--gamma', default=0.75, type=float,
                    metavar='GAMMA', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--milestones', default='10,15', type=str,
                    metavar='MIL', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--min-softmax-epoch', type=int, default=40, metavar='MINEPOCH',
                    help='minimum epoch for initial parameter using softmax (default: 2')
parser.add_argument('--veri-pairs', type=int, default=12800, metavar='VP',
                    help='number of epochs to train (default: 10)')

# Training options
# Model options
parser.add_argument('--model', type=str, help='path to voxceleb1 test dataset')
parser.add_argument('--resnet-size', default=8, type=int,
                    metavar='RES', help='The channels of convs layers)')
parser.add_argument('--inst-norm', action='store_true', default=False,
                    help='replace batchnorm with instance norm')
parser.add_argument('--channels', default='64,128,256', type=str,
                    metavar='CHA', help='The channels of convs layers)')
parser.add_argument('--feat-dim', default=161, type=int, metavar='FEAT',
                    help='acoustic feature dimension')
parser.add_argument('--remove-vad', action='store_true', default=False,
                    help='using Cosine similarity')

parser.add_argument('--alpha', default=12, type=float, metavar='FEAT',
                    help='acoustic feature dimension')
parser.add_argument('--kernel-size', default='5,5', type=str, metavar='KE',
                    help='kernel size of conv filters')
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--avg-size', type=int, default=4, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--embedding-size-a', type=int, default=128, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--embedding-size-b', type=int, default=64, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--embedding-size-o', type=int, default=32, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--input-per-spks', type=int, default=224, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--num-valid', type=int, default=5, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-input-per-file', type=int, default=4, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-batch-size', type=int, default=4, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--dropout-p', type=float, default=0., metavar='BST',
                    help='input batch size for testing (default: 64)')

# loss configure
parser.add_argument('--loss-type', type=str, default='soft', choices=['soft', 'asoft', 'center', 'amsoft'],
                    help='path to voxceleb1 test dataset')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='using Cosine similarity')
parser.add_argument('--loss-ratio', type=float, default=0.1, metavar='LOSSRATIO',
                    help='the ratio softmax loss - triplet loss (default: 2.0')
parser.add_argument('--dom-ratio', type=float, default=0.1, metavar='DOMAINLOSSRATIO',
                    help='the ratio softmax loss - triplet loss (default: 2.0')

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
parser.add_argument('--lambda-max', type=float, default=1000, metavar='S',
                    help='random seed (default: 0)')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=0, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    metavar='WEI', help='weight decay (default: 0.0)')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='MOM', help='momentum for sgd (default: 0.9)')
parser.add_argument('--dampening', default=0, type=float,
                    metavar='DAM', help='dampening for sgd (default: 0.0)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')

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
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.multiprocessing.set_sharing_strategy('file_system')

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

# create logger Define visulaize SummaryWriter instance
writer = SummaryWriter(logdir=args.check_path, filename_suffix='_first')

sys.stdout = NewLogger(osp.join(args.check_path, 'log.txt'))

kwargs = {'num_workers': args.nj, 'pin_memory': False} if args.cuda else {}
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
        concateinputfromMFB(num_frames=c.NUM_FRAMES_SPECT, remove_vad=args.remove_vad),
        # varLengthFeat(),
        to2tensor()
    ])
    transform_T = transforms.Compose([
        concateinputfromMFB(num_frames=c.NUM_FRAMES_SPECT, input_per_file=args.test_input_per_file,
                            remove_vad=args.remove_vad),
        # varLengthFeat(),
        to2tensor()
    ])
    transform_V = transforms.Compose([
        varLengthFeat(remove_vad=args.remove_vad),
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
torch.multiprocessing.set_sharing_strategy('file_system')

if args.feat_format == 'kaldi':
    file_loader = read_mat
elif args.feat_format == 'npy':
    file_loader = np.load

train_dir = ScriptTrainDataset(dir=args.train_dir, samples_per_speaker=args.input_per_spks, loader=file_loader,
                               transform=transform, num_valid=args.num_valid, domain=args.domain)
test_dir = ScriptTestDataset(dir=args.test_dir, loader=file_loader, transform=transform_T)

if len(test_dir) < args.veri_pairs:
    args.veri_pairs = len(test_dir)
    print('There are %d verification pairs.' % len(test_dir))
else:
    test_dir.partition(args.veri_pairs)

valid_dir = ScriptValidDataset(valid_set=train_dir.valid_set, loader=file_loader, spk_to_idx=train_dir.spk_to_idx,
                               dom_to_idx=train_dir.dom_to_idx, valid_utt2dom_dict=train_dir.valid_utt2dom_dict,
                               valid_uid2feat=train_dir.valid_uid2feat, valid_utt2spk_dict=train_dir.valid_utt2spk_dict,
                               transform=transform, domain=args.domain)


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # test_display_triplet_distance = False
    # print the experiment configuration
    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))
    print('Number of Speakers: {}.\n'.format(train_dir.num_spks))

    # instantiate model and initialize weights
    kernel_size = args.kernel_size.split(',')
    kernel_size = [int(x) for x in kernel_size]
    padding = [int((x - 1) / 2) for x in kernel_size]

    kernel_size = tuple(kernel_size)
    padding = tuple(padding)

    channels = args.channels.split(',')
    channels = [int(x) for x in channels]

    model_kwargs = {'embedding_size_a': args.embedding_size_a,
                    'embedding_size_b': args.embedding_size_b,
                    'embedding_size_o': args.embedding_size_o,
                    'inst_norm': args.inst_norm,
                    'resnet_size': args.resnet_size,
                    'num_classes_a': train_dir.num_spks,
                    'num_classes_b': train_dir.num_doms,
                    'channels': channels,
                    'avg_size': args.avg_size,
                    'alpha': args.alpha,
                    'kernel_size': kernel_size,
                    'padding': padding,
                    'dropout_p': args.dropout_p}

    print('Model options: {}'.format(model_kwargs))
    model = create_model(args.model, **model_kwargs)

    start_epoch = 0
    if args.save_init and not args.finetune:
        check_path = '{}/checkpoint_{}.pth'.format(args.check_path, start_epoch)
        torch.save(model, check_path)

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']

            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
            model_dict = model.state_dict()
            model_dict.update(filtered)
            model.load_state_dict(model_dict)
            #
            # model.dropout.p = args.dropout_p
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    ce_criterion = nn.CrossEntropyLoss()
    if args.loss_type == 'soft':
        xe_criterion = None
    elif args.loss_type == 'asoft':
        ce_criterion = None
        model.classifier_spk = AngleLinear(in_features=args.embedding_size, out_features=train_dir.num_spks, m=args.m)
        xe_criterion = AngleSoftmaxLoss(lambda_min=args.lambda_min, lambda_max=args.lambda_max)
    elif args.loss_type == 'center':
        xe_criterion = CenterLoss(num_classes=train_dir.num_spks, feat_dim=args.embedding_size)
    elif args.loss_type == 'amsoft':
        ce_criterion = None
        model.classifier_spk = AdditiveMarginLinear(feat_dim=args.embedding_size, n_classes=train_dir.num_spks)
        xe_criterion = AMSoftmaxLoss(margin=args.margin, s=args.s)

    optimizer = create_optimizer(model.parameters(), args.optimizer, **opt_kwargs)
    if args.loss_type == 'center':
        optimizer = torch.optim.SGD([{'params': xe_criterion.parameters(), 'lr': args.lr * 5},
                                     {'params': model.parameters()}],
                                    lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    if args.finetune:
        if args.loss_type == 'asoft' or args.loss_type == 'amsoft':
            classifier_params = list(map(id, model.classifier.parameters()))
            rest_params = filter(lambda p: id(p) not in classifier_params, model.parameters())
            optimizer = torch.optim.SGD([{'params': model.classifier.parameters(), 'lr': args.lr * 5},
                                         {'params': rest_params}],
                                        lr=args.lr, weight_decay=args.weight_decay,
                                        momentum=args.momentum)

    if args.scheduler == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    else:
        milestones = args.milestones.split(',')
        milestones = [int(x) for x in milestones]
        milestones.sort()
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    ce = [ce_criterion, xe_criterion]

    start = args.start_epoch + start_epoch
    print('Start epoch is : ' + str(start))
    # start = 0
    end = start + args.epochs

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(args.batch_size / 2), shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    # sitw_test_loader = torch.utils.data.DataLoader(sitw_test_dir, batch_size=args.test_batch_size,
    #                                                shuffle=False, **kwargs)
    # sitw_dev_loader = torch.utils.data.DataLoader(sitw_dev_part, batch_size=args.test_batch_size, shuffle=False,
    #                                               **kwargs)

    if args.cuda:
        model = model.cuda()
        for i in range(len(ce)):
            if ce[i] != None:
                ce[i] = ce[i].cuda()
        print('Dropout is {}.'.format(model.dropout_p))

    for epoch in range(start, end):
        # pdb.set_trace()
        print('\n\33[1;34m Current \'{}\' learning rate is '.format(args.optimizer), end='')
        for param_group in optimizer.param_groups:
            print('{:.5f} '.format(param_group['lr']), end='')
        print(' \33[0m')

        train(train_loader, model, ce, optimizer, epoch)
        if epoch % 4 == 1 or epoch == (end - 1):
            check_path = '{}/checkpoint_{}.pth'.format(args.check_path, epoch)
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'criterion': ce},
                       check_path)

        if epoch % 2 == 1 and epoch != (end - 1):
            test(test_loader, valid_loader, model, epoch)

        scheduler.step()

        # exit(1)

    extract_dir = KaldiExtractDataset(dir=args.test_dir, transform=transform_V, filer_loader=file_loader)
    extract_loader = torch.utils.data.DataLoader(extract_dir, batch_size=1, shuffle=False, **kwargs)
    xvector_dir = args.check_path
    xvector_dir = xvector_dir.replace('checkpoint', 'xvector')
    verification_extract(extract_loader, model, xvector_dir)

    verify_dir = ScriptVerifyDataset(dir=args.test_dir, trials_file=args.trials, xvectors_dir=xvector_dir,
                                     loader=read_vec_flt)
    verify_loader = torch.utils.data.DataLoader(verify_dir, batch_size=64, shuffle=False, **kwargs)
    verification_test(test_loader=verify_loader, dist_type=('cos' if args.cos_sim else 'l2'),
                      log_interval=args.log_interval)

    writer.close()


def train(train_loader, model, ce, optimizer, epoch):
    # switch to evaluate mode
    model.train()

    correct_a = 0.
    correct_b = 0.

    total_datasize = 0.
    total_loss_a = 0.
    total_loss_b = 0.
    total_loss = 0.
    # for param_group in optimizer.param_groups:
    #     print('\33[1;34m Optimizer \'{}\' learning rate is {}.\33[0m'.format(args.optimizer, param_group['lr']))
    ce_criterion, xe_criterion = ce
    pbar = tqdm(enumerate(train_loader))
    output_softmax = nn.Softmax(dim=1)

    for batch_idx, (data, label_a, label_b) in pbar:

        if args.cuda:
            data = data.cuda()

        data, label_a = Variable(data), Variable(label_a)
        label_b = Variable(label_b)

        logits_spk, feat_spk, logits_dom, feat_dom = model(data)

        true_labels_a = label_a.cuda()
        true_labels_b = label_b.cuda()

        # pdb.set_trace()

        # cos_theta, phi_theta = classfier
        spk_label = logits_spk
        dom_lable = logits_dom

        if args.loss_type == 'soft':
            spk_loss = ce_criterion(logits_spk, true_labels_a)

        elif args.loss_type == 'asoft':
            spk_label, _ = spk_label
            spk_loss = xe_criterion(logits_spk, true_labels_a)
        elif args.loss_type == 'center':
            loss_cent = ce_criterion(logits_spk, true_labels_a)
            loss_xent = xe_criterion(feat_spk, true_labels_a)
            spk_loss = args.loss_ratio * loss_xent + loss_cent
        elif args.loss_type == 'amsoft':
            spk_loss = xe_criterion(logits_spk, true_labels_a)

        dom_loss = (args.dom_ratio * ce_criterion(dom_lable, true_labels_b))
        loss = spk_loss + dom_loss

        predicted_labels_a = output_softmax(spk_label)

        predicted_one_labels_a = torch.max(predicted_labels_a, dim=1)[1]
        minibatch_correct_a = float((predicted_one_labels_a.cuda() == true_labels_a.cuda()).sum().item())
        minibatch_acc_a = minibatch_correct_a / len(predicted_one_labels_a)
        correct_a += minibatch_correct_a

        predicted_labels_b = output_softmax(dom_lable)
        predicted_one_labels_b = torch.max(predicted_labels_b, dim=1)[1]
        minibatch_correct_b = float((predicted_one_labels_b.cuda() == true_labels_b.cuda()).sum().item())
        minibatch_acc_b = minibatch_correct_b / len(predicted_one_labels_b)
        correct_b += minibatch_correct_b

        total_datasize += len(predicted_one_labels_a)
        total_loss_a += float(spk_loss.item())
        total_loss_b += float(dom_loss.item())
        total_loss += float(loss.item())

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()

        if args.loss_type == 'center' and args.loss_ratio != 0:
            for param in xe_criterion.parameters():
                param.grad.data *= (1. / args.loss_ratio)

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch {:2d}: [{:8d}/{:8d} ({:3.0f}%)] Avg Loss: {:.4f} Spk Loss: {:.4f} Dom Loss: {:.4f} ' \
                'Batch: Spk Accuracy: {:.4f}%, Dom Accuracy: {:.4f}%'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    total_loss / (batch_idx + 1),
                    total_loss_a / (batch_idx + 1),
                    total_loss_b / (batch_idx + 1),
                    100. * minibatch_acc_a,
                    100. * minibatch_acc_b))

    print('\n\33[91mTrain Epoch {}: Avg loss: {:.4f} Spk Loss: {:.4f} Dom Loss: {:.4f} .'.format(epoch,
                                                                                                 total_loss / len(
                                                                                                     train_loader),
                                                                                                 total_loss_a / len(
                                                                                                     train_loader),
                                                                                                 total_loss_b / len(
                                                                                                     train_loader)))

    print('Spk Accuracy:{:.4f}%, Dom Accuracy:{:.4f}%.\33[0m'.format(100 * correct_a / total_datasize,
                                                                     100 * correct_b / total_datasize, ))

    writer.add_scalar('Train/Spk_Accuracy', correct_a / total_datasize, epoch)
    writer.add_scalar('Train/Dom_Accuracy', correct_b / total_datasize, epoch)
    writer.add_scalar('Train/Loss', total_loss / len(train_loader), epoch)

    torch.cuda.empty_cache()


def test(test_loader, valid_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    valid_pbar = tqdm(enumerate(valid_loader))
    softmax = nn.Softmax(dim=1)

    correct_a = 0.
    correct_b = 0.

    total_datasize = 0.

    for batch_idx, (data, label_a, label_b) in valid_pbar:
        data = Variable(data.cuda())

        # compute output
        out_a, _, out_b, _ = model(data)
        if args.loss_type == 'asoft':
            predicted_labels_a, _ = out_a

        else:
            predicted_labels_a = out_a
        predicted_labels_b = out_b

        true_labels_a = Variable(label_a.cuda())
        true_labels_b = Variable(label_b.cuda())

        # pdb.set_trace()
        predicted_one_labels_a = softmax(predicted_labels_a)
        predicted_one_labels_a = torch.max(predicted_one_labels_a, dim=1)[1]

        batch_correct_a = (predicted_one_labels_a.cuda() == true_labels_a.cuda()).sum().item()
        minibatch_acc_a = float(batch_correct_a / len(predicted_one_labels_a))
        correct_a += batch_correct_a

        predicted_one_labels_b = softmax(predicted_labels_b)
        predicted_one_labels_b = torch.max(predicted_one_labels_b, dim=1)[1]
        batch_correct_b = (predicted_one_labels_b.cuda() == true_labels_b.cuda()).sum().item()
        minibatch_acc_b = float(batch_correct_b / len(predicted_one_labels_b))
        correct_b += batch_correct_b

        total_datasize += len(predicted_one_labels_a)

        if batch_idx % args.log_interval == 0:
            valid_pbar.set_description(
                'Valid Epoch: {:2d} [{:8d}/{:8d} ({:3.0f}%)] Batch Spk Accuracy: {:.4f}% Dom Accuracy: {:.4f}%'.format(
                    epoch,
                    batch_idx * len(data),
                    len(valid_loader.dataset),
                    100. * batch_idx / len(valid_loader),
                    100. * minibatch_acc_a,
                    100. * minibatch_acc_b
                ))

    spk_valid_accuracy = 100. * correct_a / total_datasize
    dom_valid_accuracy = 100. * correct_b / total_datasize

    writer.add_scalar('Test/Spk_Valid_Accuracy', spk_valid_accuracy, epoch)
    writer.add_scalar('Test/Dom_Valid_Accuracy', dom_valid_accuracy, epoch)

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
        _, out_a_, _, _ = model(data_a)
        _, out_p_, _, _ = model(data_p)
        out_a = out_a_
        out_p = out_p_

        dists = l2_dist.forward(out_a, out_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        dists = dists.reshape(vec_shape[0], vec_shape[1]).mean(dim=1)
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

    mindcf_01, mindcf_001 = evaluate_kaldi_mindcf(distances, labels)
    writer.add_scalar('Test/mindcf-0.01', mindcf_01, epoch)
    writer.add_scalar('Test/mindcf-0.001', mindcf_001, epoch)

    dist_type = 'cos' if args.cos_sim else 'l2'
    print('\nFor %s_distance, ' % dist_type)
    print('  \33[91mTest Spk ERR is {:.4f}%, Threshold is {}'.format(100. * eer, eer_threshold))
    print('  mindcf-0.01 {:.4f}, mindcf-0.001 {:.4f},'.format(mindcf_01, mindcf_001))
    print('  Valid Spk Accuracy is %.4f %%, Dom Accuracy is %.4f %% .\33[0m' % (spk_valid_accuracy, dom_valid_accuracy))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
