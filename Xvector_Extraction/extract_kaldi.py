#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: extract_kaldi.py
@Time: 2019/12/10 下午10:32
@Overview: Exctract speakers vectors for kaldi PLDA.
"""
from __future__ import print_function
import argparse
import pathlib
import pdb
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os

import numpy as np
from tqdm import tqdm
from Define_Model.ResNet import ResNet
from Process_Data.VoxcelebTestset import VoxcelebTestset
# from Process_Data.voxceleb2_wav_reader import voxceleb2_list_reader
from Process_Data.KaldiDataset import write_xvector_ark
from eval_metrics import evaluate_kaldi_eer

from Process_Data.DeepSpeakerDataset_dynamic import ClassificationDataset, ValidationDataset, SampleTrainDataset
from Process_Data.voxceleb_wav_reader import wav_list_reader, test_list_reader, wav_duration_reader

from Define_Model.model import PairwiseDistance, SuperficialResCNN
from Process_Data.audio_processing import GenerateSpect, concateinputfromMFB, varLengthFeat, PadCollate, ExtractCollate
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
import warnings

warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Model options
# parser.add_argument('--dataroot', type=str, default='Data/dataset/voxceleb1/Fbank64_Norm',
#                     help='path to dataset')
parser.add_argument('--dataroot', type=str, default='Data/dataset/voxceleb1/spect_161',
                    help='path to voxceleb1 test dataset')

parser.add_argument('--test-pairs-path', type=str, default='Data/dataset/voxceleb1/test_trials/ver_list.txt',
                    help='path to pairs file')

# parser.add_argument('--ckp-dir', type=str, default='Data/checkpoint/ResNet10/Fb_No',
#                     help='folder to output model checkpoints')
parser.add_argument('--ckp-dir', type=str, default='Data/checkpoint/SuResCNN10/soft',
                    help='folder to output model checkpoints')
parser.add_argument('--epoch', type=int, default=19,
                    help='epoch of checkpoint to load model')

# Training options
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=1, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

# parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
parser.add_argument('--n-triplets', type=int, default=100000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=0.1, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--loss-ratio', type=float, default=2.0, metavar='LOSSRATIO',
                    help='the ratio softmax loss - triplet loss (default: 2.0')

parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=0, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
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

# create logger
# Define visulaize SummaryWriter instance

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}


if args.cos_sim:
    l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6)
else:
    l2_dist = PairwiseDistance(2)

# voxceleb, voxceleb_dev = wav_list_reader(args.test_dataroot)
voxceleb, train_set, valid_set = wav_duration_reader(data_path=args.dataroot)
voxceleb_test = test_list_reader(args.dataroot)

if args.acoustic_feature == 'fbank':
    transform = transforms.Compose([
        # truncatedinputfromMFB(),
        #concateinputfromMFB(),
        varLengthFeat(),
        totensor()
    ])
    transform_T = transforms.Compose([
        # truncatedinputfromMFB(input_per_file=args.test_input_per_file),
        # concateinputfromMFB(input_per_file=args.test_input_per_file),
        varLengthFeat(),
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

# train_dir = SampleTrainDataset(vox_duration=train_set, dir=args.dataroot, loader=file_loader, transform=transform, return_uid=True)
train_dir = ClassificationDataset(voxceleb=train_set, dir=args.dataroot, loader=file_loader, transform=transform, return_uid=True)
test_dir = ClassificationDataset(voxceleb=voxceleb_test, dir=args.dataroot, loader=file_loader, transform=transform, return_uid=True)


del voxceleb
del train_set
del valid_set

def extract(train_loader, model, dataset, extract_path=args.ckp_dir + '/kaldi_feat'):

    model.eval()
    uids, xvector = [], torch.Tensor([])
    pbar = tqdm(enumerate(train_loader))

    for batch_idx, (data, label, uid) in pbar:

        data = Variable(data.cuda())
        # print(data.shape)
        # pdb.set_trace()
        # feats = model.pre_forward(data)
        _, feats = model(data)
        feats = feats.data.cpu()

        xvector = torch.cat((xvector, feats), dim=0)

        for i in range(len(uid)):
            uids.append(uid[i])

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Extract {}: [{:8d}/{:8d} ({:3.0f}%)] '.format(
                    dataset,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)))

    np_xvector = xvector.numpy().astype(np.float32)
    write_xvector_ark(uids, np_xvector, write_path=extract_path, set=dataset)

    # make utt2spk files

    #


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive

    # print the experiment configuration
    print('\33[91m\nCurrent time is {}.\33[0m'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))
    print('Number of Speakers: {}\n'.format(len(train_dir.classes)))

    # instantiate model and initialize weights
    # model = ResNet(layers=[1, 1, 1, 1],
    #                channels=[64, 128, 256, 512],
    #                embedding=args.embedding_size,
    #                num_classes=len(train_dir.classes),
    #                expansion=2)
    model = SuperficialResCNN(layers=[1, 1, 1, 0], embedding_size=args.embedding_size, n_classes=1211, m=3)

    if args.cuda:
        model.cuda()
    # optionally resume from a checkpoint
    resume = args.ckp_dir + '/checkpoint_{}.pth'.format(args.epoch)

    if os.path.isfile(resume):
        print('=> loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume)
        filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
        model.load_state_dict(filtered)

    else:
        raise Exception('=> no checkpoint found at {}'.format(args.resume))

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # Extract Train set vectors
    extract(train_loader, model, dataset='train')

    # Extract test set vectors
    extract(test_loader, model, dataset='test')





if __name__ == '__main__':
    main()

