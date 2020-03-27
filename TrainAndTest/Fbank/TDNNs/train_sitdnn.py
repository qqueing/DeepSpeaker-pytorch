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

import time

import torch.nn as nn
import torch
import pathlib
import os
import numpy as np
from tensorboardX import SummaryWriter
from torch import optim

from torch.autograd import Variable
from torchvision import transforms
# import trainset
# import testset
from tqdm import tqdm

from Define_Model.TDNN import Time_Delay
from torch.utils.data import DataLoader
import warnings
import argparse
import pdb

from Process_Data.DeepSpeakerDataset_dynamic import ClassificationDataset
from Process_Data.VoxcelebTestset import VoxcelebTestset
from Process_Data.audio_processing import truncatedinputfromMFB, totensor, read_MFB, make_Fbank, conver_to_wav, \
    concateinputfromMFB, PadCollate
from Process_Data.voxceleb2_wav_reader import voxceleb2_list_reader
from Process_Data import constants as c
from Process_Data.voxceleb_wav_reader import wav_list_reader
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

parser.add_argument('--dataroot', type=str, default='/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/fbank64',
                    help='path to local dataset')
parser.add_argument('--test-dataroot', type=str, default='/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/fbank64',
                    help='path to local dataset')
parser.add_argument('--dataset', type=str, default='/home/cca01/work2019/Data/voxceleb2',
                    help='path to dataset')
parser.add_argument('--test-pairs-path', type=str, default='Data/dataset/ver_list.txt',
                    help='path to pairs file')
parser.add_argument('--check-path', type=str, default='Data/checkpoint/tdnn_vox1/ada',
                    help='path to dataset')
parser.add_argument('--resume', default='Data/checkpoint/tdnn_vox1/ada/checkpoint_24.pth',
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
parser.add_argument('--epochs', type=int, default=45, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--make-feats', action='store_true', default=False,
                    help='need to make spectrograms file')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='adagrad', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')

parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument('--gpu-id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
# net = Time_Delay(context, 24, 300, node_num, full_context)
# net = net.to(device)

writer = SummaryWriter(logdir='Log/tdnn/ada', filename_suffix='fb64_vox1')
voxceleb, train_set, valid_set = wav_list_reader(args.dataroot, split=True)

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
        totensor()
    ])
    file_loader = read_MFB

if args.make_feats:
    num_pro = 0.
    skip_wav = 0.
    for datum in voxceleb:
        # Data/voxceleb1/
        # /data/voxceleb/voxceleb1_wav/
        # pdb.set_trace()
        filename = '/home/cca01/work2019/Data/voxceleb2/' + datum['filename'] + '.wav'
        write_path = args.dataroot + '/' + datum['filename'] + '.npy'

        if os.path.exists(write_path):
            num_pro += 1
            skip_wav += 1
            continue

        if os.path.exists(filename):
            make_Fbank(filename=filename, write_path=write_path)
            num_pro += 1
        # convert the audio format for m4a.
        elif os.path.exists(filename.replace('.wav', '.m4a')):
            conver_to_wav(filename.replace('.wav', '.m4a'),
                          write_path=args.dataroot + '/' + datum['filename'] + '.wav')

            make_Fbank(filename=args.dataroot + '/' + datum['filename'] + '.wav',
                       write_path=write_path,
                       nfilt=c.TDNN_FBANK_FILTER,
                       use_energy=True)
            num_pro += 1
        else:
            raise ValueError(filename + ' doesn\'t exist.')

        print(
            '\tPreparing for speaker {}. \tProcessed {:2f}% {}/{}. \tSkipped {} wav files.'.format(datum['speaker_id'],
                                                                                                   100 * num_pro / len(
                                                                                                       voxceleb),
                                                                                                   num_pro,
                                                                                                   len(voxceleb),
                                                                                                   skip_wav), end='\r')

    print('\nComputing Fbank features success! Skipped %d wav files.' % skip_wav)
    exit(1)

if args.cos_sim:
    l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6)
else:
    l2_dist = nn.PairwiseDistance(p=2)

train_dir = ClassificationDataset(voxceleb=train_set, dir=args.dataroot, loader=file_loader, transform=transform)
test_dir = VoxcelebTestset(dir=args.dataroot, pairs_path=args.test_pairs_path, loader=file_loader,
                           transform=transform_T)
valid_dir = ClassificationDataset(voxceleb=valid_set, dir=args.dataroot, loader=file_loader, transform=transform)

del voxceleb
del train_set
del valid_set


def train(train_loader, model, optimizer, epoch):
    # switch to evaluate mode
    model.train()

    running_loss = 0.
    total = 0
    correct = 0.
    ce_loss = nn.CrossEntropyLoss()
    output_softmax = nn.Softmax(dim=1)

    # learning rate multiple 0.1 per 15 epochs
    for param_group in optimizer.param_groups:
        # if param_group['lr']==0.01:
        # param_group['lr']=0.1
        # pdb.set_trace()
        print('\33[1;34m Current \'{}\' learning rate is {}.\33[0m'.format(args.optimizer, param_group['lr']))

    pbar = tqdm(enumerate(train_loader))
    # pdb.set_trace()

    for batch_idx, (data, label) in pbar:
        # break
        data, label = Variable(data), Variable(label)
        data = data.cuda()
        label = label.cuda()
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
        print(
            '\33[1;34m For train epoch {}, global accuracy is {}%, and average of batch loss: {:.4f}.\33[0m \n'.format(
                epoch, 100. * float(correct) / total, running_loss / len(train_loader)))
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
    # testloader = DataLoader(test_set, batch_size=128, num_workers=16, shuffle=True)
    # torch.set_num_threads(16)

    pbar = tqdm(enumerate(test_loader))
    # .set_trace()

    for batch_idx, (data_a, data_p, label) in pbar:
        current_sample = data_a.size(0)
        data_a = data_a.resize_(args.test_input_per_file * current_sample, 1, data_a.size(2), data_a.size(3))
        data_p = data_p.resize_(args.test_input_per_file * current_sample, 1, data_a.size(2), data_a.size(3))
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        # compute output
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

    if args.cos_sim:
        print(
            '\33[91mFor cos_distance, Test set ERR is {:.4f}, threshold is {}. Valid accuracy could be {:.2f}%.\n\33[0m'.format(
                100. * eer, eer_threshold, valid_accuracy))
    else:
        print(
            '\33[91mFor l2_distance, Test set ERR: {:.8f}%\tBest ACC:{:.8f}\n\33[0m'.format(100. * eer, valid_accuracy))


def main():
    # print the experiment configuration
    print('\33[91m \nCurrent time is {}\33[0m'.format(str(time.asctime())))
    # print('Parsed options:\n{}\n'.format(vars(args)))
    print('Number of Speakers: {}\n'.format(len(train_dir.classes)))

    context = [[-2, 2], [-2, 0, 2], [-3, 0, 3], [0], [0]]
    # the same configure as x-vector
    node_num = [64, 128, 256, 512, 1024, 1024, 512, 512]
    full_context = [True, False, False, True, True]

    # train_set = trainset.TrainSet('../all_feature/')
    # todo:
    # train_set = []
    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size,
                                               collate_fn=PadCollate(dim=2),
                                               shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=args.test_batch_size,
                                               ollate_fn=PadCollate(dim=2),
                                               shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Time_Delay(context, 64, len(train_dir.classes), node_num, full_context)

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)
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
        train(train_loader, model, optimizer, epoch)
        test(test_loader, valid_loader, model, epoch)


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.99, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(),
                                   lr=new_lr,
                                   lr_decay=args.lr_decay,
                                   weight_decay=args.wd)
    return optimizer


if __name__ == '__main__':
    main()
