#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: sires_extract_model.py
@Time: 2020/3/24 2:36 PM
@Overview:
"""
from __future__ import print_function
import argparse
import pathlib
import pdb
import pickle
import random
import time
import torch
import torch.nn as nn
from Process_Data import constants as c
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
from Define_Model.ResNet import SimpleResNet

from kaldi_io import read_mat
import numpy as np
from tqdm import tqdm
from Process_Data.KaldiDataset import ScriptTrainDataset, ScriptTestDataset, ScriptValidDataset, SitwTestDataset
from Define_Model.model import PairwiseDistance
from Process_Data.audio_processing import concateinputfromMFB, PadCollate, varLengthFeat, to2tensor
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
parser.add_argument('--train-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/dev_no_sil',
                    help='path to dataset')
parser.add_argument('--test-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/test_no_sil',
                    help='path to voxceleb1 test dataset')

parser.add_argument('--sitw-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/sitw_spect',
                    help='path to voxceleb1 test dataset')

parser.add_argument('--check-path', default='Data/checkpoint/SiResNet34/soft/kaldi',
                    help='folder to output model checkpoints')
parser.add_argument('--extract-path', default='Data/extract/SiResNet34/soft/kaldi',
                    help='folder to output model checkpoints')

# Training options
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=128, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--sample-utt', type=int, default=120, metavar='ES',
                    help='Dimensionality of the embedding')
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

if args.cuda:
    cudnn.benchmark = True

# Define visulaize SummaryWriter instance
kwargs = {'num_workers': 12, 'pin_memory': True} if args.cuda else {}
l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else PairwiseDistance(2)

transform = transforms.Compose([
    concateinputfromMFB(num_frames=c.MINIMUIN_LENGTH, remove_vad=True),
    # varLengthFeat(),
    to2tensor()
])
transform_T = transforms.Compose([
    concateinputfromMFB(num_frames=c.MINIMUIN_LENGTH, input_per_file=args.test_input_per_file, remove_vad=True),
    # varLengthFeat(),
    to2tensor()
])

train_dir = ScriptTrainDataset(dir=args.train_dir, samples_per_speaker=args.input_per_spks, transform=transform,
                               return_uid=True, loader=read_mat)
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


def model_extract(model, epoch):
    # switch to evaluate mode
    model.eval()

    file_dir = args.extract_path + '/epoch_%d' % epoch
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if args.cuda:
        model_conv1 = model.conv1.weight.cpu().detach().numpy()
        np.save(file_dir + '/model.conv1.npy', model_conv1)

        model_fc1_weight = model.fc1.weight.cpu().detach().numpy()
        np.save(file_dir + '/model.fc1.weight.npy', model_fc1_weight)

        model_fc1_bias = model.fc1.bias.cpu().detach().numpy()
        np.save(file_dir + '/model.fc1.bias.npy', model_fc1_bias)

    print('Saving pairs in %s.\n' % file_dir)


# def test_extract(test_loader, model, epoch, set_name):
#     # switch to evaluate mode
#     model.eval()
#
#     utt_con = []
#     pbar = tqdm(enumerate(test_loader))
#     save_per_num = 10
#     for batch_idx, (data_a, data_b, label, uid_a, uid_b) in pbar:
#         # pdb.set_trace()
#         data = torch.cat((data_a,data_b), dim=0)
#         uid = torch.cat((uid_a, uid_b), dim=0)
#
#         data = Variable(data.cuda(), requires_grad=True)
#
#         conv1 = model.conv1(data)
#         bn1 = model.bn1(conv1)
#         relu1 = model.relu(bn1)
#         logit, _ = model(data)
#         cos_theta, phi_theta = logit
#         cos_theta[0][label.long()].backward()
#
#         grad = data.grad.cpu().numpy().squeeze().astype(np.float32)
#         conv1 = conv1.cpu().detach().numpy().squeeze().astype(np.float32)
#         bn1 = bn1.cpu().detach().numpy().squeeze().astype(np.float32)
#         relu1 = relu1.cpu().detach().numpy().squeeze().astype(np.float32)
#
#         utt_con.append((uid, conv1, bn1, relu1, grad))
#         if batch_idx % args.log_interval == 0:
#             pbar.set_description('Saving output for {} : [{:8d}/{:8d} ({:3.0f}%)] '.format(
#                 uid,
#                 batch_idx * len(data),
#                 len(test_loader.dataset),
#                 100. * batch_idx / len(test_loader)))
#
#         if (batch_idx + 1) % save_per_num == 0 or (batch_idx + 1) == len(test_loader.dataset):
#             num = batch_idx // save_per_num if batch_idx + 1 % save_per_num == 0 else batch_idx // save_per_num + 1
#             # checkpoint_dir / extract / < dataset > / < set >.*.bin
#
#             filename = args.check_path + '/extract/epoch_%d/%s.%d.bin' % (epoch, set_name, num)
#             print('Saving pairs in %s.' % filename)
#
#             file_path = pathlib.Path(filename)
#             if not file_path.parent.exists():
#                 os.makedirs(str(file_path.parent))
#
#             with open(filename, 'wb') as f:
#                 pickle.dump(utt_con, f)
#
#             utt_con = []

def main():
    class_to_idx = train_dir.spk_to_idx
    # class_to_idx = np.load('Data/dataset/voxceleb1/Fbank64_Norm/class2idx.npy').item()
    print('\nNumber of Speakers: {}.'.format(len(class_to_idx)))
    # print the experiment configuration
    print('Current time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))

    # instantiate model and initialize weights
    model = SimpleResNet(layers=[3, 4, 6, 3], num_classes=len(class_to_idx))

    resume_path = args.check_path + '/checkpoint_{}.pth'
    epochs = np.arange(0, 31)

    for e in epochs:
        # Load model from Checkpoint file
        if os.path.isfile(resume_path.format(e)):
            print('=> loading checkpoint {}'.format(resume_path.format(e)))
            checkpoint = torch.load(resume_path.format(e))
            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
            model.load_state_dict(filtered)
        else:
            print('=> no checkpoint found at %s' % resume_path.format(e))
            continue

        model_extract(model, e)


if __name__ == '__main__':
    main()
