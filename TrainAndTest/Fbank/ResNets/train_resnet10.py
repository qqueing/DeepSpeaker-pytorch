"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: test_accuracy.py
@Time: 19-11-21 下午21:12
@Overview: Train the resnet 10 with ce.
"""
from __future__ import print_function
import argparse
import pathlib
import pdb
import time

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os

import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from Define_Model.ResNet import ResNet
from Process_Data.VoxcelebTestset import VoxcelebTestset
# from Process_Data.voxceleb2_wav_reader import voxceleb2_list_reader
from eval_metrics import evaluate_kaldi_eer

from logger import Logger

from Process_Data.DeepSpeakerDataset_dynamic import ClassificationDataset, ValidationDataset
from Process_Data.voxceleb_wav_reader import wav_list_reader

from Define_Model.model import PairwiseDistance
from Process_Data.audio_processing import concateinputfromMFB, varLengthFeat, PadCollate
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, truncatedinputfromMFB, read_MFB, read_audio
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

import warnings

warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')

# Dataset options
parser.add_argument('--dataroot', type=str, default='/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/Fbank64_Norm',
                    help='path to dataset')
parser.add_argument('--test-dataroot', type=str,
                    default='/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/Fbank64_Norm',
                    help='path to voxceleb1 test dataset')
parser.add_argument('--test-pairs-path', type=str, default='Data/dataset/ver_list.txt',
                    help='path to pairs file')

# Checkpoint path options
parser.add_argument('--ckp-dir', default='Data/checkpoint/ResNet10/Fb_No',
                    help='folder to output model checkpoints')
parser.add_argument('--resume',
                    default='Data/checkpoint/ResNet10/Fb_No/checkpoint_20.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Epoch options
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=30, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--min-softmax-epoch', type=int, default=20, metavar='MINEPOCH',
                    help='minimum epoch for initial parameter using softmax (default: 2')

# Training options
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

parser.add_argument('--margin', type=float, default=0.1, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--loss-ratio', type=float, default=2.0, metavar='LOSSRATIO',
                    help='the ratio softmax loss - triplet loss (default: 2.0')

# optimizer options
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=0, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='W', help='momentum (default: 0.9)')
parser.add_argument('--dampening', default=0, type=float,
                    metavar='W', help='dampening (default: 0.0)')
parser.add_argument('--optimizer', default='adagrad', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=3, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=15, metavar='LI',
                    help='how many batches to wait before logging training status')

# Making acoustic features
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

# Define visulaize SummaryWriter instance
writer = SummaryWriter(logdir=args.ckp_dir, filename_suffix='varlen')

# Args for dataloader
kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

if args.cos_sim:
    l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6)
else:
    l2_dist = PairwiseDistance(2)

# voxceleb, voxceleb_dev = wav_list_reader(args.test_dataroot)
voxceleb, train_set, valid_set = wav_list_reader(args.dataroot, split=True)

# voxceleb2, voxceleb2_dev = voxceleb2_list_reader(args.dataroot)

# if args.makemfb:
#     #pbar = tqdm(voxceleb)
#     for datum in voxceleb:
#         mk_MFB((args.dataroot +'/voxceleb1_wav/' + datum['filename']+'.wav'))
#     print("Complete convert")
#
# if args.makespec:
#     num_pro = 1.
#     for datum in voxceleb:
#         # Data/voxceleb1/
#         # /data/voxceleb/voxceleb1_wav/
#         GenerateSpect(wav_path='/data/voxceleb/voxceleb1_wav/' + datum['filename']+'.wav',
#                       write_path=args.dataroot +'/spectrogram/voxceleb1_wav/' + datum['filename']+'.npy')
#         print('\rprocessed {:2f}% {}/{}.'.format(num_pro/len(voxceleb), num_pro, len(voxceleb)), end='\r')
#         num_pro += 1
#     print('\nComputing Spectrograms success!')
#     exit(1)

if args.acoustic_feature == 'fbank':
    transform = transforms.Compose([
        # truncatedinputfromMFB(),
        # concateinputfromMFB(),
        varLengthFeat(),
        totensor()
    ])
    transform_T = transforms.Compose([
        # truncatedinputfromMFB(input_per_file=args.test_input_per_file),
        concateinputfromMFB(input_per_file=args.test_input_per_file),
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

# pdb.set_trace()

train_dir = ClassificationDataset(voxceleb=train_set, dir=args.dataroot, loader=file_loader, transform=transform)
test_dir = VoxcelebTestset(dir=args.dataroot, pairs_path=args.test_pairs_path, loader=file_loader,
                           transform=transform_T)
valid_dir = ValidationDataset(voxceleb=valid_set, dir=args.dataroot, loader=file_loader,
                              class_to_idx=train_dir.class_to_idx, transform=transform)

del voxceleb
del train_set
del valid_set


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    test_display_triplet_distance = False

    # print the experiment configuration
    print('\33[91m\nCurrent time is {}.\33[0m'.format(str(time.asctime())))
    print('Parsed options: {}\n'.format(vars(args)))
    print('Number of Speakers: {}\n'.format(len(train_dir.classes)))

    # instantiate model and initialize weights
    model = ResNet(layers=[1, 1, 1, 1],
                   channels=[64, 128, 256, 512],
                   embedding=args.embedding_size,
                   num_classes=len(train_dir.classes),
                   expansion=2)

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[20, 80], gamma=0.1)
    # criterion = AngularSoftmax(in_feats=args.embedding_size,
    #                           num_classes=len(train_dir.classes))

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
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                print('No scheduler!')
            # criterion.load_state_dict(checkpoint['criterion'])

        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    print('Start epoch is : ' + str(start) + '\n')
    # start = 0
    end = start + args.epochs

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=PadCollate(dim=2), **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=args.test_batch_size, shuffle=False,
                                               collate_fn=PadCollate(dim=2), **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    for epoch in range(start, end):
        # pdb.set_trace()
        train(train_loader, model, optimizer, scheduler, epoch)
        test(test_loader, valid_loader, model, epoch)
        scheduler.step()
        # exit(1)

    writer.close()


def train(train_loader, model, optimizer, scheduler, epoch):
    # switch to evaluate mode
    model.train()
    # labels, distances = [], []
    correct = 0.
    total_datasize = 0.
    total_loss = 0.

    for param_group in optimizer.param_groups:
        print('\33[1;34m Optimizer \'{}\' learning rate is {}.\33[0m'.format(args.optimizer, param_group['lr']))

    pbar = tqdm(enumerate(train_loader))
    # pdb.set_trace()
    ce = nn.CrossEntropyLoss()
    output_softmax = nn.Softmax(dim=1)

    for batch_idx, (data, label) in pbar:
        if args.cuda:
            data = data.cuda()
        data, label = Variable(data), Variable(label)

        # pdb.set_trace()
        feats = model.pre_forward(data)
        classfier = model(feats)

        predicted_labels = output_softmax(classfier)
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]
        true_labels = label.cuda()

        cross_entropy_loss = ce(classfier, true_labels)
        loss = cross_entropy_loss  # + triplet_loss * args.loss_ratio

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
                'Train Epoch: {:2d} [{:8d}/{:8d} ({:3.0f}%)] Loss: {:.6f} Batch Accuracy: {:.6f}%'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                    100. * minibatch_acc))

    check_path = pathlib.Path('{}/checkpoint_{}.pth'.format(args.ckp_dir, epoch))
    if not check_path.parent.exists():
        os.makedirs(str(check_path.parent))

    torch.save({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
               str(check_path))

    print('\n\33[91mFor epoch {}: Train set Accuracy:{:.6f}%, and Average loss is {}. \n\33[0m'.format(epoch,
                                                                                                       100 * float(
                                                                                                           correct) / total_datasize,
                                                                                                       total_loss / len(
                                                                                                           train_loader)))
    writer.add_scalar('Train/Accuracy', correct / total_datasize, epoch)
    writer.add_scalar('Train/Loss', total_loss / len(train_loader), epoch)


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
                'Test Epoch for Classification: {:2d} [{:8d}/{:8d} ({:3.0f}%)] Batch Accuracy: {:.4f}%'.format(
                    epoch,
                    batch_idx * len(data),
                    len(valid_loader.dataset),
                    100. * batch_idx / len(valid_loader),
                    100. * minibatch_acc
                ))

    valid_accuracy = 100. * correct / total_datasize
    writer.add_scalar('Test/Valid_Accuracy', valid_accuracy, epoch)

    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        current_sample = data_a.size(0)
        data_a = data_a.resize_(args.test_input_per_file * current_sample, 1, data_a.size(2), data_a.size(3))
        data_p = data_p.resize_(args.test_input_per_file * current_sample, 1, data_a.size(2), data_a.size(3))

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a), Variable(data_p), Variable(label)

        # compute output
        out_a, out_p = model.pre_forward(data_a), model.pre_forward(data_p)

        dists = l2_dist.forward(out_a, out_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        dists = dists.data.cpu().numpy()
        dists = dists.reshape(current_sample, args.test_input_per_file).mean(axis=1)
        distances.append(dists)
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset), 100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    writer.add_scalar('Test/EER', eer, epoch)
    writer.add_scalar('Test/Threshold', eer_threshold, epoch)

    if args.cos_sim:
        print(
            '\33[91mFor cos_distance, Test set verification ERR is {:.8f}%, when threshold is {}. Valid set classificaton accuracy is {:.2f}%.\n\33[0m'.format(
                100. * eer, eer_threshold, valid_accuracy))
    else:
        print(
            '\33[91mFor l2_distance, Test set verification ERR is {:.8f}%, when threshold is {}. Valid set classificaton accuracy is {:.2f}%.\n\33[0m'.format(
                100. * eer, eer_threshold, valid_accuracy))


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=args.momentum, dampening=args.dampening,
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
