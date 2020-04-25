#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: extract_xvector_kaldi.py
@Time: 2019/12/10 下午10:32
@Overview: Exctract speakers vectors for kaldi PLDA.
"""
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
import torch._utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from kaldi_io import read_mat
from torch.autograd import Variable
from tqdm import tqdm

from Define_Model.model import PairwiseDistance
# from Process_Data.voxceleb2_wav_reader import voxceleb2_list_reader
from Process_Data.KaldiDataset import write_vec_ark, \
    KaldiExtractDataset
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, read_audio
from Process_Data.audio_processing import varLengthFeat
from TrainAndTest.common_func import create_model

# Version conflict

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
parser = argparse.ArgumentParser(description='Extract x-vector for plda')
# Model options
parser.add_argument('--train-dir', type=str, help='path to dataset')
parser.add_argument('--test-dir', type=str, help='path to voxceleb1 test dataset')
parser.add_argument('--sitw-dir', type=str, help='path to voxceleb1 test dataset')

parser.add_argument('--check-path', help='folder to output model checkpoints')
parser.add_argument('--extract-path', help='folder to output model grads, etc')

# Model options
parser.add_argument('--model', type=str, choices=['LoResNet10', 'ResNet20', 'SiResNet34', 'SuResCNN10'],
                    help='path to voxceleb1 test dataset')
parser.add_argument('--feat-dim', default=24, type=int, metavar='FEAT',
                    help='acoustic feature dimension')
parser.add_argument('--dropout-p', type=float, default=0., metavar='BST',
                    help='model global dropout p)')
parser.add_argument('--epoch', type=int, default=36, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--batch-size', type=int, default=1, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='1', type=str,
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

kwargs = {'num_workers': 12, 'pin_memory': True} if args.cuda else {}

l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else PairwiseDistance(2)

if args.acoustic_feature == 'fbank':
    transform = transforms.Compose([
        varLengthFeat(),
        totensor()
    ])
    transform_T = transforms.Compose([
        varLengthFeat(),
        totensor()
    ])
    file_loader = read_mat
else:
    transform = transforms.Compose([
        truncatedinput(),
        toMFB(),
        totensor(),
        # tonormal()
    ])
    file_loader = read_audio

# pdb.set_trace()
train_dir = KaldiExtractDataset(dir=args.train_dir,
                                loader=file_loader,
                                transform=transform)

test_dir = KaldiExtractDataset(dir=args.test_dir,
                               loader=file_loader,
                               transform=transform)


# test_dir = ScriptTestDataset(dir=args.test_dir, loader=file_loader, transform=transform_T)


def extract(data_loader, model, set_id, extract_path):

    model.eval()
    uids, xvector = [], torch.Tensor([])
    pbar = tqdm(enumerate(data_loader))

    for batch_idx, (data, label, uid) in pbar:
        data = Variable(data.cuda())
        _, feats = model(data)
        feats = feats.data.cpu()
        feats = feats.squeeze()

        xvector = torch.cat((xvector, feats), dim=0)

        for i in range(len(uid)):
            uids.append(uid[i])

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Extract {}: [{:8d}/{:8d} ({:3.0f}%)] '.format(
                    set_id,
                    batch_idx * len(data),
                    len(data_loader.dataset),
                    100. * batch_idx / len(data_loader)))

    np_xvector = xvector.numpy().astype(np.float32)
    write_vec_ark(uid=uids, feats=np_xvector,
                  write_path=extract_path, set_id=set_id)


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive

    # print the experiment configuration
    print('\nCurrent time is\33[91m {}\33[0m.'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))
    print('Number of Speakers: {}\n'.format(len(train_dir.classes)))

    # instantiate model and initialize weights
    model_kwargs = {'input_dim': args.feat_dim,
                    'embedding_size': args.embedding_size,
                    'num_classes': args.num_classes,
                    'dropout_p': args.dropout_p}

    print('Model options: {}'.format(model_kwargs))

    model = create_model(args.model, **model_kwargs)

    if args.cuda:
        model.cuda()
    # optionally resume from a checkpoint
    resume = args.ckp_dir + '/checkpoint_{}.pth'.format(args.epoch)

    if os.path.isfile(resume):
        print('=> loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume)
        filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
        model_dict = model.state_dict()
        model_dict.update(filtered)

        model.load_state_dict(model_dict)

    else:
        raise Exception('=> no checkpoint found at {}'.format(resume))

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.batch_size, shuffle=False, **kwargs)

    # Extract Train set vectors
    extract(train_loader, model, dataset='train', extract_path=args.extract_path + '/x_vector')

    # Extract test set vectors
    extract(test_loader, model, dataset='test', extract_path=args.extract_path + '/x_vector')

    print('Extract x-vector completed for train and test!\n')





if __name__ == '__main__':
    main()

