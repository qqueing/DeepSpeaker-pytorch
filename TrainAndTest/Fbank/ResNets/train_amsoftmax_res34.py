#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: test_accuracy.py
@Time: 19-8-6 下午18:02
@Overview: Train the resnet 34 with am-softmax.
"""
# from __future__ import print_function
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os

import numpy as np
from tqdm import tqdm
from Define_Model import ResSpeakerModel
from Process_Data.VoxcelebTestset import VoxcelebTestset
from eval_metrics import evaluate_kaldi_eer

from logger import Logger

from Process_Data.DeepSpeakerDataset_dynamic import ClassificationDataset
from Process_Data.voxceleb_wav_reader import wav_list_reader

from Define_Model import PairwiseDistance
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, truncatedinputfromMFB, read_MFB, read_audio, \
    mk_MFB
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
# Model options
parser.add_argument('--dataroot', type=str, default='Data/dataset',
                    help='path to dataset')
parser.add_argument('--test-pairs-path', type=str, default='Data/dataset/ver_list.txt',
                    help='path to pairs file')

parser.add_argument('--log-dir', default='data/pytorch_speaker_logs',
                    help='folder to output model checkpoints')

parser.add_argument('--ckp-dir', default='Data/checkpoint',
                    help='folder to output model checkpoints')

parser.add_argument('--resume',
                    default='Data/checkpoint/resnet34_amsoftmax/checkpoint_1.pth',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=45, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=8, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

# parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
parser.add_argument('--n-triplets', type=int, default=100000, metavar='N',
                    help='how many triplets will generate from the dataset')

parser.add_argument('--margin', type=float, default=0.1, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')

parser.add_argument('--min-softmax-epoch', type=int, default=2, metavar='MINEPOCH',
                    help='minimum epoch for initial parameter using softmax (default: 2')

parser.add_argument('--loss-ratio', type=float, default=2.0, metavar='LOSSRATIO',
                    help='the ratio softmax loss - triplet loss (default: 2.0')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=1, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument('--mfb', action='store_true', default=True,
                    help='start from MFB file')
parser.add_argument('--makemfb', action='store_true', default=False,
                    help='need to make mfb file')

args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.cuda:
    cudnn.benchmark = True
CKP_DIR = args.ckp_dir
LOG_DIR = args.log_dir + '/run-test_{}-n{}-lr{}-wd{}-m{}-embeddings{}-msceleb-alpha10' \
    .format(args.optimizer, args.n_triplets, args.lr, args.wd,
            args.margin, args.embedding_size)

# create logger
logger = Logger(LOG_DIR)
# Define visulaize SummaryWriter instance
writer = SummaryWriter('Log/amsoftmax_resnet34')

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
if args.cos_sim:
    l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6)
else:
    l2_dist = PairwiseDistance(2)

voxceleb, voxceleb_dev = wav_list_reader(args.dataroot)
if args.makemfb:
    # pbar = tqdm(voxceleb)
    for datum in voxceleb:
        mk_MFB((args.dataroot + '/voxceleb1_wav/' + datum['filename'] + '.wav'))
    print("Complete convert")

if args.mfb:
    transform = transforms.Compose([
        truncatedinputfromMFB(),
        totensor()
    ])
    transform_T = transforms.Compose([
        truncatedinputfromMFB(input_per_file=args.test_input_per_file),
        totensor()
    ])
    file_loader = read_MFB
else:
    transform = transforms.Compose([
        truncatedinput(),
        toMFB(),
        totensor(),
        # tonormal()
    ])
    file_loader = read_audio

train_dir = ClassificationDataset(voxceleb=voxceleb_dev, dir=args.dataroot, loader=file_loader, transform=transform)
test_dir = VoxcelebTestset(dir=args.dataroot, pairs_path=args.test_pairs_path, loader=file_loader,
                           transform=transform_T)

del voxceleb
del voxceleb_dev


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    test_display_triplet_distance = False

    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    print('\nNumber of Classes:\n{}\n'.format(len(train_dir.classes)))

    # instantiate model and initialize weights
    model = ResSpeakerModel(embedding_size=args.embedding_size, resnet_size=34, num_classes=len(train_dir.classes))

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)

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

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    for epoch in range(start, end):
        # pdb.set_trace()
        train(train_loader, model, optimizer, epoch)
        test(test_loader, model, epoch)

    writer.close()


def train(train_loader, model, optimizer, epoch):
    # switch to evaluate mode
    model.train()
    # labels, distances = [], []
    correct = 0.
    total_datasize = 0.
    total_loss = 0.

    pbar = tqdm(enumerate(train_loader))
    # pdb.set_trace()

    for batch_idx, (data, label) in pbar:
        if args.cuda:
            data = data.cuda()
        data, label = Variable(data), Variable(label)

        # pdb.set_trace()
        out = model(data)
        classifier_out = model.forward_classifier(out)

        output_softmax = nn.Softmax()
        predicted_labels = output_softmax(classifier_out)
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]

        true_labels = label.cuda()

        loss = model.AMSoftmaxLoss(out, true_labels.cuda())
        # loss = cross_entropy_loss  # + triplet_loss * args.loss_ratio

        minibatch_acc = float((predicted_one_labels.cuda() == true_labels.cuda()).sum().data[0]) / len(
            predicted_one_labels)
        correct += float((predicted_one_labels.cuda() == true_labels.cuda()).sum().data[0])
        total_datasize += len(predicted_one_labels)
        total_loss += loss.data[0]
        # Visualize loss and acc
        writer.add_scalar('Train_Loss/epoch_%d' % epoch, loss.data[0], batch_idx)
        writer.add_scalar('Train_Accuracy/epoch_%d' % epoch, minibatch_acc, batch_idx)
        # pdb.set_trace()

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f} \tMinibatch Accuracy: {:.6f}%'.format(
                    epoch,
                    batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0],
                    100. * minibatch_acc))

    torch.save({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'criterion': criterion.state_dict()
                },
               '{}/resnet34_amsoftmax/checkpoint_{}.pth'.format(CKP_DIR, epoch))

    print('\33[91mFor AMSoftmax Train set Accuracy:{:.6f}% \n\33[0m'.format(100 * float(correct) / total_datasize))
    writer.add_scalar('Train_Accuracy_Per_Epoch', correct / total_datasize, epoch)
    writer.add_scalar('Train_Loss_Per_Epoch', total_loss / len(train_loader), epoch)


def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        current_sample = data_a.size(0)
        data_a = data_a.resize_(args.test_input_per_file * current_sample, 1, data_a.size(2), data_a.size(3))
        data_p = data_p.resize_(args.test_input_per_file * current_sample, 1, data_a.size(2), data_a.size(3))
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        # compute output
        out_a, out_p = model(data_a), model(data_p)

        dists = l2_dist.forward(out_a, out_p)
        dists = dists.data.cpu().numpy()
        dists = dists.reshape(current_sample, args.test_input_per_file).mean(axis=1)
        distances.append(dists)
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    # err, accuracy= evaluate_eer(distances,labels)
    eer, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim)
    writer.add_scalar('Test_Result/eer', eer, epoch)
    writer.add_scalar('Test_Result/accuracy', accuracy, epoch)
    # tpr, fpr, accuracy, val, far = evaluate(distances, labels)

    if args.cos_sim:
        print('\33[91mFor cos_distance Test set: ERR: {:.8f}%\tBest ACC:{:.8f} \n\33[0m'.format(100. * eer,
                                                                                                np.mean(accuracy)))
    else:
        print('\33[91mFor l2_distance Test set: ERR: {:.8f}%\tBest ACC:{:.8f} \n\33[0m'.format(100. * eer,
                                                                                               np.mean(accuracy)))

    # logger.log_value('Test Accuracy', np.mean(accuracy))


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
    return optimizer


if __name__ == '__main__':
    main()
