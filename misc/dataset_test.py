#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: dataset_test.py
@Time: 2019/12/17 9:18 PM
@Overview:
"""
import argparse
import pdb
import numpy as np
from torchvision import transforms


from Process_Data.DeepSpeakerDataset_dynamic import SampleTrainDataset
from Process_Data.audio_processing import concateinputfromMFB, totensor, read_MFB
from Process_Data.voxceleb_wav_reader import wav_list_reader, wav_duration_reader

parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')

# Dataset options
parser.add_argument('--dataroot', type=str, default='/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/spect_161',
                    help='path to dataset')
parser.add_argument('--test-dataroot', type=str, default='/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/spect_161',
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

transform = transforms.Compose([
    # truncatedinputfromMFB(),
    concateinputfromMFB(),
    totensor()
])
transform_T = transforms.Compose([
    # truncatedinputfromMFB(input_per_file=args.test_input_per_file),
    concateinputfromMFB(input_per_file=args.test_input_per_file),
    totensor()
])
file_loader = read_MFB



# voxceleb, train_set, valid_set = wav_list_reader(args.dataroot, split=True)
voxceleb, train_set, valid_set = wav_duration_reader(data_path=args.dataroot)
# train_set = np.load(args.dataroot + '/vox_duration.npy')

train_dir = SampleTrainDataset(vox_duration=train_set, dir=args.dataroot, loader=file_loader, transform=transform)

pdb.set_trace()
print(len(train_dir))