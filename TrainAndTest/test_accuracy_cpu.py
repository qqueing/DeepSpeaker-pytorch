#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: test_accuracy.py
@Time: 19-6-19 下午5:17
@Overview:
"""
#from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
import pdb

import numpy as np
from tqdm import tqdm
from Define_Model import DeepSpeakerModel
from eval_metrics import evaluate_eer
from logger import Logger

#from DeepSpeakerDataset_static import DeepSpeakerDataset
from Process_Data.VoxcelebTestset import VoxcelebTestset

from Define_Model import PairwiseDistance
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, truncatedinputfromMFB,read_MFB,read_audio
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

parser.add_argument('--resume', default='Data/checkpoint/checkpoint_16.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=29, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--batch-size', type=int, default=256, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=8, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

#parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
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
parser.add_argument('--optimizer', default='adagrad', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=True,
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
LOG_DIR = args.log_dir + '/run-test_{}-n{}-lr{}-wd{}-m{}-embeddings{}-msceleb-alpha10'\
    .format(args.optimizer, args.n_triplets, args.lr, args.wd,
            args.margin,args.embedding_size)

# create logger
logger = Logger(LOG_DIR)


kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
l2_dist = PairwiseDistance(2)


# voxceleb = read_my_voxceleb_structure(args.dataroot)
# if args.makemfb:
    #pbar = tqdm(voxceleb)
    # for datum in voxceleb:
    #     # print(datum['filename'])
    #     mk_MFB((args.dataroot +'/voxceleb1_wav/' + datum['filename']+'.wav'))
    # print("Complete convert")

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
                        #tonormal()
                    ])
    file_loader = read_audio


# voxceleb_dev = [datum for datum in voxceleb if datum['subset']=='dev']
# train_dir = DeepSpeakerDataset(voxceleb = voxceleb_dev, dir=args.dataroot,n_triplets=args.n_triplets,loader = file_loader,transform=transform)
# del voxceleb
# del voxceleb_dev

test_dir = VoxcelebTestset(dir=args.dataroot, pairs_path=args.test_pairs_path, loader=file_loader, transform=transform_T)

qwer = test_dir.__getitem__(3)


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    test_display_triplet_distance = False

    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    print('\nNumber of Test utterance pair:\n{}\n'.format(len(test_dir.validation_images)))

    # instantiate model and initialize weights
    model = DeepSpeakerModel(embedding_size=args.embedding_size, resnet_size=10, num_classes=1211)

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            #checkpoint = torch.load(args.resume, map_location='cpu')

            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}

            model.load_state_dict(filtered)
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    # train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    epoch = args.start_epoch
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    #for epoch in range(start, end):

    test(test_loader, model, epoch)
        #break;

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
        pdb.set_trace()
        dists = l2_dist.forward(out_a, out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        dists = dists.data.cpu().numpy()
        dists = dists.reshape(current_sample,args.test_input_per_file).mean(axis=1)
        distances.append(dists)
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    #print("distance {.8f}".format(distances))
    #print("distance {.1f}".format(labels))
    tpr, fpr, fnr, accuracy, val,  far = evaluate_eer(distances,labels)
    print('\33[91mTest set: Accuracy: {:.8f} EER:{:.8f} \n\33[0m'.format(np.mean(accuracy), fnr))
    logger.log_value('Test Accuracy', np.mean(accuracy))


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
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

