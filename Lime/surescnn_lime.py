#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: surescnn_lime.py
@Time: 2019/12/17 3:57 PM
@Overview:
"""
from __future__ import print_function
import argparse
import pathlib
import pdb
import random
import time
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
from Define_Model.SoftmaxLoss import AngleSoftmaxLoss
from Process_Data.VoxcelebTestset import VoxcelebTestset
from eval_metrics import evaluate_kaldi_eer

from Process_Data.DeepSpeakerDataset_dynamic import ClassificationDataset, ValidationDataset
from Process_Data.voxceleb_wav_reader import wav_list_reader

from Define_Model.model import PairwiseDistance, SuperficialResCNN
from Process_Data.audio_processing import concateinputfromMFB, PadCollate, varLengthFeat
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, truncatedinputfromMFB, read_MFB, read_audio, mk_MFB
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
parser.add_argument('--dataroot', type=str, default='/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/spect_161',
                    help='path to dataset')
parser.add_argument('--test-dataroot', type=str, default='/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/spect_161',
                    help='path to voxceleb1 test dataset')
parser.add_argument('--test-pairs-path', type=str, default='Data/dataset/ver_list.txt',
                    help='path to pairs file')

parser.add_argument('--log-dir', default='data/pytorch_speaker_logs',
                    help='folder to output model checkpoints')
parser.add_argument('--ckp-dir', default='Data/checkpoint/SuResCNN10/spect/ada_trans',
                    help='folder to output model checkpoints')
parser.add_argument('--resume',
                    default='Data/checkpoint/SuResCNN10/spect/ada_trans/checkpoint_27.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Training options
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=2, metavar='S',
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

# Define visulaize SummaryWriter instance
writer = SummaryWriter(logdir=args.ckp_dir, filename_suffix='sgd_0.1')

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

if args.cos_sim:
    l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6)
else:
    l2_dist = PairwiseDistance(2)

# voxceleb, voxceleb_dev = wav_list_reader(args.test_dataroot)

transform = transforms.Compose([
    concateinputfromMFB(),
    totensor()
])
transform_T = transforms.Compose([
    concateinputfromMFB(input_per_file=args.test_input_per_file),
    totensor()
])
file_loader = read_MFB


# pdb.set_trace()

# train_dir = ClassificationDataset(voxceleb=train_set, dir=args.dataroot, loader=file_loader, transform=transform)

# class_to_idx = np.load('Data/dataset/voxceleb1/Fbank64_Norm/class2idx.npy').item()
# print('Number of Speakers: {}.\n'.format(len(class_to_idx)))
#
# test_dir = VoxcelebTestset(dir=args.dataroot, pairs_path=args.test_pairs_path, loader=file_loader, transform=transform_T)
#
# indices = list(range(len(test_dir)))
# random.shuffle(indices)
# indices = indices[:4000]
# test_part = torch.utils.data.Subset(test_dir, indices)
#
# valid_dir = ValidationDataset(voxceleb=valid_set, dir=args.dataroot, loader=file_loader, class_to_idx=class_to_idx, transform=transform)

# try:
#     np.save('Data/dataset/voxceleb1/Fbank64_Norm/class2idx.npy', class_to_idx)
#     print('Saving Success!')
#     exit(1)
# except:
#     print('Saving Error!')

def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    voxceleb, train_set, valid_set = wav_list_reader(args.dataroot, split=True)

    class_to_idx = np.load('Data/dataset/voxceleb1/Fbank64_Norm/class2idx.npy').item()
    print('Number of Speakers: {}.\n'.format(len(class_to_idx)))

    test_dir = VoxcelebTestset(dir=args.dataroot, pairs_path=args.test_pairs_path, loader=file_loader,
                               transform=transform_T)

    indices = list(range(len(test_dir)))
    random.shuffle(indices)
    indices = indices[:4000]
    test_part = torch.utils.data.Subset(test_dir, indices)

    valid_dir = ValidationDataset(voxceleb=valid_set, dir=args.dataroot, loader=file_loader, class_to_idx=class_to_idx,
                                  transform=transform)

    del voxceleb
    del train_set
    del valid_set

    # print the experiment configuration
    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))

    # instantiate model and initialize weights
    model = SuperficialResCNN(layers=[1, 1, 1, 0], embedding_size=args.embedding_size, n_classes=len(class_to_idx), m=3)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
            model.load_state_dict(filtered)

        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    # train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=True,
    #                                            # collate_fn=PadCollate(dim=2),
    #                                            **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=args.test_batch_size, shuffle=False,
                                               # collate_fn=PadCollate(dim=2),
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(test_part, batch_size=args.test_batch_size, shuffle=False, **kwargs)



    # train(train_loader, model, ce, optimizer, scheduler, epoch)
    # train(train_loader, model, ce)
    # test(test_loader, valid_loader, model)

    model.eval()

    valid_pbar = tqdm(enumerate(valid_loader))
    softmax = nn.Softmax(dim=1)

    correct = 0.
    total_datasize = 0.

    for batch_idx, (data, label) in valid_pbar:
        # data = Variable(data.cuda())
        # true_labels = Variable(label.cuda())
        pdb.set_trace()
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(data,
                                                 batch_predict,  # classification function
                                                 top_labels=5,
                                                 hide_color=0,
                                                 num_samples=len(data))

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                    hide_rest=False)
        img_boundry2 = mark_boundaries(temp / 255.0, mask)
        plt.imshow(img_boundry2)

        break

    writer.close()

def train(train_loader, model, ce, optimizer, scheduler, epoch):
    # switch to evaluate mode
    model.eval()
    # labels, distances = [], []
    correct = 0.
    total_datasize = 0.
    total_loss = 0.

    for param_group in optimizer.param_groups:
        print('\33[1;34m Optimizer \'{}\' learning rate is {}.\33[0m'.format(args.optimizer, param_group['lr']))

    pbar = tqdm(enumerate(train_loader))
    #pdb.set_trace()

    output_softmax = nn.Softmax(dim=1)

    for batch_idx, (data, label) in pbar:
        if args.cuda:
            data = data.cuda()

        data, label = Variable(data), Variable(label)

        # pdb.set_trace()
        classfier, _ = model(data)
        cos_theta, phi_theta = classfier

        probs = F.softmax(cos_theta, dim=1)

        true_labels = label.cuda()



        predicted_labels = output_softmax(cos_theta)
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]
        minibatch_acc = float((predicted_one_labels.cuda() == true_labels.cuda()).sum().item()) / len(predicted_one_labels)
        correct += float((predicted_one_labels.cuda()==true_labels.cuda()).sum().item())
        total_datasize += len(predicted_one_labels)


        if batch_idx % args.log_interval == 0:
            pbar.set_description('Train Epoch: {:2d} [{:8d}/{:8d} ({:3.0f}%)] Batch Accuracy: {:.6f}%'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                100. * minibatch_acc))

    check_path = pathlib.Path('{}/checkpoint_{}.pth'.format(args.ckp_dir, epoch))
    if not check_path.parent.exists():
        os.makedirs(str(check_path.parent))

    torch.save({'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
                #'criterion': criterion.state_dict()
               str(check_path))

    print('\33[91mFor epoch {}: Train set Accuracy:{:.6f}%, and Average loss is {}.\n\33[0m'.format(epoch, 100 * float(correct) / total_datasize, total_loss/len(train_loader)))
    writer.add_scalar('Train/Accuracy', correct/total_datasize, epoch)
    writer.add_scalar('Train/Loss', total_loss/len(train_loader), epoch)

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
        out, _ = model(data)
        cos_theta, phi_theta = out
        predicted_labels = cos_theta
        true_labels = Variable(label.cuda())

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(data,
                                                 batch_predict,  # classification function
                                                 top_labels=5,
                                                 hide_color=0,
                                                 num_samples=len(data))

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

    valid_accuracy = 100. * correct/total_datasize
    writer.add_scalar('Test/Valid_Accuracy', valid_accuracy, epoch)

    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        current_sample = data_a.size(0)
        data_a = data_a.resize_(args.test_input_per_file *current_sample, 1, data_a.size(2), data_a.size(3))
        data_p = data_p.resize_(args.test_input_per_file *current_sample, 1, data_a.size(2), data_a.size(3))

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a), Variable(data_p), Variable(label)

        # compute output
        _, out_a_ = model(data_a)
        _, out_p_ = model(data_p)
        out_a = out_a_
        out_p = out_p_


        dists = l2_dist.forward(out_a, out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
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
        print('\33[91mFor l2_distance, Test set verification ERR is {:.8f}%, when threshold is {}. Valid set classificaton accuracy is {:.2f}%.\n\33[0m'.format(
                100. * eer, eer_threshold, valid_accuracy))

def batch_predict(model, batch):
    model.eval()
    # batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    if args.cuda:
        model = model.cuda()
        batch = batch.cuda()

    out, _ = model(batch)
    logits, phi_theta = out

    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


if __name__ == '__main__':
    main()

