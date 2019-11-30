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
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os


import numpy as np
from tqdm import tqdm

from Define_Model.ResNet import SimpleResNet
from Define_Model.TDNN import Time_Delay
from Define_Model.model import ResSpeakerModel
from eval_metrics import evaluate_kaldi_eer

from logger import Logger

#from DeepSpeakerDataset_static import DeepSpeakerDataset
from Process_Data.DeepSpeakerDataset_dynamic import DeepSpeakerDataset, ClassificationDataset
from Process_Data.VoxcelebTestset import VoxcelebTestset
from Process_Data.voxceleb_wav_reader import wav_list_reader

from Define_Model.model import PairwiseDistance, ResCNNSpeaker, SuperficialResNet
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, truncatedinputfromMFB, read_MFB, read_audio, \
    mk_MFB, concateinputfromMFB
# Version conflict

import warnings
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
# Model options
parser.add_argument('--dataroot', type=str, default='Data/dataset/voxceleb1/fbank64',
                    help='path to dataset')
parser.add_argument('--test-pairs-path', type=str, default='Data/dataset/ver_list.txt',
                    help='path to pairs file')

parser.add_argument('--log-dir', default='data/pytorch_speaker_logs',
                    help='folder to output model checkpoints')

parser.add_argument('--ckp-dir', default='Data/checkpoint',
                    help='folder to output model checkpoints')

# parser.add_argument('--resume',
#                     default='Data/checkpoint/resnet34_asoftmax/checkpoint_26.pth', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
#
# parser.add_argument('--resume',
#                     default='Data/checkpoint/resnet34_asoftmax/checkpoint_26.pth', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')

# parser.add_argument('--resume',
#                     default='Data/checkpoint/sures10_asoft/checkpoint_6.pth',
#                     type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')

# parser.add_argument('--resume',
#                     default='Data/checkpoint/resnet34_asoftmax/7.7%/checkpoint_{}.pth',
#                     type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')

parser.add_argument('--resume',
                    default='Data/checkpoint/10res_soft/1013_10/checkpoint_{}.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# parser.add_argument('--resume',
#                     default='Data/checkpoint/resnet10_asoftmax/checkpoint_%d.pth', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')

# parser.add_argument('--resume',
#                     default='Data/checkpoint/sires34_soft/checkpoint_{}.pth', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')


# parser.add_argument('--resume',
#                     default='Data/checkpoint/tdnn_vox1/checkpoint_25.pth',
#                     type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')

parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=1, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--resnet-size', type=int, default=10, metavar='E',
                    help='depth of resnet to train (default: 34)')
parser.add_argument('--batch-size', type=int, default=512, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

#parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
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
parser.add_argument('--optimizer', default='adagrad', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=3, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=1, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument('--mfb', action='store_true', default=True,
                    help='start from MFB file')
parser.add_argument('--makemfb', action='store_true', default=False,
                    help='need to make mfb file')

parser.add_argument('--acoustic-feature', choices=['fbank', 'spectrogram', 'mfcc'], default='fbank',
                    help='choose the acoustic features type.')
parser.add_argument('--makespec', action='store_true', default=False,
                    help='need to make spectrograms file')

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
LOG_DIR = args.log_dir + '/run-test{}-lr{}-wd{}-m{}-embeddings{}'.format(args.optimizer, args.lr, args.wd, args.margin, args.embedding_size)

# create logger
logger = Logger(LOG_DIR)


kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
if args.cos_sim:
    l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6)
else:
    l2_dist = PairwiseDistance(2)

voxceleb, voxceleb_dev = wav_list_reader(args.dataroot)

if args.makemfb:
    #pbar = tqdm(voxceleb)
    for datum in voxceleb:
        mk_MFB((args.dataroot +'/voxceleb1_wav/' + datum['filename']+'.wav'))
    print("Complete convert")

# if args.mfb:
#     transform = transforms.Compose([
#         truncatedinputfromMFB(),
#         totensor()
#     ])
#     transform_T = transforms.Compose([
#         truncatedinputfromMFB(input_per_file=args.test_input_per_file),
#         totensor()
#     ])
#     file_loader = read_MFB
# else:
#     transform = transforms.Compose([
#                         truncatedinput(),
#                         toMFB(),
#                         totensor(),
#                         #tonormal()
#                     ])
#     file_loader = read_audio
if args.acoustic_feature=='fbank':
    transform = transforms.Compose([
        concateinputfromMFB(),
        #truncatedinputfromMFB(),
        totensor()
    ])
    transform_T = transforms.Compose([
        # truncatedinputfromMFB(input_per_file=args.test_input_per_file),
        concateinputfromMFB(input_per_file=args.test_input_per_file),
        totensor()
    ])
    file_loader = read_MFB

elif args.acoustic_feature=='spectrogram':
    # Start from spectrogram
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


train_dir = ClassificationDataset(voxceleb=voxceleb_dev, dir=args.dataroot, loader=file_loader, transform=transform)

del voxceleb
del voxceleb_dev

test_dir = VoxcelebTestset(dir=args.dataroot, pairs_path=args.test_pairs_path, loader=file_loader, transform=transform_T)
writer = SummaryWriter('Log/softmax_res10', filename_suffix='1106')

# qwer = test_dir.__getitem__(3)


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # test_display_triplet_distance = False

    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    print('\nNumber of Classes:\n{}\n'.format(len(train_dir.classes)))

    # instantiate model and initialize weights
    # initial different models
    # model = ResSpeakerModel(embedding_size=args.embedding_size,
    #                          resnet_size=args.resnet_size,
    #                          num_classes=len(train_dir.classes))
    #
    model = ResCNNSpeaker(embedding_size=args.embedding_size,
                             resnet_size=args.resnet_size,
                             num_classes=len(train_dir.classes))

    # model = SuperficialResNet(layers=[1, 1, 1, 1],
    #                           embedding_size=args.embedding_size,
    #                           n_classes=1211,
    #                           m=3)

    # model = ResSpeakerModel(embedding_size=args.embedding_size,
    #                         resnet_size=10,
    #                         num_classes=len(train_dir.classes))

    # TDNN
    # context = [[-2, 2], [-2, 0, 2], [-3, 0, 3], [0], [0]]
    # # the same configure as x-vector
    # node_num = [512, 512, 512, 512, 1500, 3000, 512, 512]
    # full_context = [True, False, False, True, True]
    #
    # model = Time_Delay(context, 64, len(train_dir.classes), node_num, full_context)

    # model = SimpleResNet(layers=[3, 4, 6, 3], num_classes=len(train_dir.classes))

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print('=> loading checkpoint {}'.format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         checkpoint = torch.load(args.resume)
    #
    #         filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
    #
    #         model.load_state_dict(filtered)
    #
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #     else:
    #         print('=> no checkpoint found at {}'.format(args.resume))

    # epoch = args.start_epoch
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.test_batch_size * 2, shuffle=False, **kwargs)

    #epochs = np.arange(8, 9)
    epochs = [8]

    for epoch in epochs:
        if os.path.isfile(args.resume.format(epoch)):
            print('=> loading checkpoint {}'.format(args.resume.format(epoch)))
            checkpoint = torch.load(args.resume.format(epoch))
            args.start_epoch = checkpoint['epoch']
            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}

            model.load_state_dict(filtered)

            # optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
            break
        # train_test(train_loader, model, epoch)
        test(test_loader, model, epoch)


def train_test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []
    x_vectors = []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, label) in pbar:

        if args.cuda:
            data_a = data_a.cuda()

        data_a = Variable(data_a)

        # compute output
        # out_a = model(data_a)

        # TDNN extract
        # out_a = model.pre_forward(data_a)
        # out_p = model.pre_forward(data_p)
        # pdb.set_trace()
        # x_vectors.append(out_a.data.cpu().numpy())
        labels.append(label.data.numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    try:
        x_vectors = np.array(x_vectors)
        labels = np.array(labels)
        # labels.append(label.numpy())
    except Exception:
        pdb.set_trace()


    # err, accuracy= evaluate_eer(distances,labels)
    # eer, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim)
    try:
        #np.save('Data/xvector/train/x_vectors.npy', x_vectors)
        np.save('Data/xvector/train/label.npy', labels)
        print('Extracted {} x_vectors from train set.'.format(len(x_vectors)))
    except:
        pdb.set_trace()

    #tpr, fpr, accuracy, val, far = evaluate(distances, labels)

    #logger.log_value('Test Accuracy', np.mean(accuracy))

def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []
    x_vectors = []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        current_sample = data_a.size(0)
        data_a = data_a.resize_(args.test_input_per_file * current_sample, 1, data_a.size(2), data_a.size(3))
        data_p = data_p.resize_(args.test_input_per_file * current_sample, 1, data_a.size(2), data_a.size(3))

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()

        data_a, data_p, label = Variable(data_a), \
                                Variable(data_p), Variable(label)

        # compute output
        out_a, out_p = model(data_a), model(data_p)

        # TDNN extract
        # out_a = model.pre_forward(data_a)
        # out_p = model.pre_forward(data_p)
        # pdb.set_trace()
        x_vectors.append((out_a.data.cpu().numpy(), out_p.data.cpu().numpy(), label.data.cpu().numpy()))

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
    try:
        x_vectors = np.array(x_vectors)
    except Exception:
        pdb.set_trace()


    # err, accuracy= evaluate_eer(distances,labels)
    # eer, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim)
    eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    writer.add_scalar('Test_Result/eer', eer, epoch)
    writer.add_scalar('Test_Result/threshold', eer_threshold, epoch)
    writer.add_scalar('Test_Result/accuracy', accuracy, epoch)
    # try:
    #     np.save('Data/xvector/test/x_vectors.npy', x_vectors)
    #     np.save('Data/xvector/test/label.npy', labels)
    # except:
    #     pdb.set_trace()

    #tpr, fpr, accuracy, val, far = evaluate(distances, labels)

    if args.cos_sim:
        print('\33[91mFor cos_distance, Test set ERR is {:.8f} when threshold is {:.8f}. And test accuracy could be {:.2f}%.\n\33[0m'.format(100. * eer, eer_threshold, 100.* accuracy))
    else:
        print('\33[91mFor l2_distance, Test set ERR is {:.8f} when threshold is {:.8f}. And test accuracy could be {:.2f}%.\n\33[0m'.format(100. * eer, eer_threshold, 100.* accuracy))

    #logger.log_value('Test Accuracy', np.mean(accuracy))


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

