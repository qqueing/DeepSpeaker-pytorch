#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: input_extract.py
@Time: 2020/3/25 5:30 PM
@Overview:
"""
import argparse
import os
import pdb
import pickle
import random
import torch

import numpy as np
import pathlib
import matplotlib.pyplot as plt
from Process_Data import constants as c
from matplotlib import animation
from scipy import interpolate
from torch import nn
import torchvision.transforms as transforms
import json
from Lime import cValue_1

from Process_Data.KaldiDataset import ScriptTrainDataset, ScriptValidDataset
from Process_Data.audio_processing import concateinputfromMFB, to2tensor

parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Model options
parser.add_argument('--train-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_aug_spect/dev',
                    help='path to dataset')
parser.add_argument('--test-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_aug_spect/test',
                    help='path to voxceleb1 test dataset')
parser.add_argument('--sitw-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/sitw_spect',
                    help='path to voxceleb1 test dataset')

parser.add_argument('--check-path', default='Data/checkpoint/SuResCNN10/spect/aug',
                    help='folder to output model checkpoints')
parser.add_argument('--extract-path', default='Lime/SuResCNN10',
                    help='folder to output model checkpoints')

# Training options
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='ES',
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
parser.add_argument('--seed', type=int, default=123456, metavar='S',
                    help='random seed (default: 0)')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Define visulaize SummaryWriter instance
kwargs = {}


def main():
    # conv1s = np.array([]).reshape((0, 64, 5, 5))
    # grads = np.array([]).reshape((0, 2, 161))
    # model_set = ['kaldi_5wd', 'aug']
    subsets = ['orignal', 'babble', 'noise', 'music', 'reverb']
    file_loader = np.load

    #
    if os.path.exists(args.extract_path + '/inputs.json'):
        # conv1s_means = np.load(args.extract_path + '/conv1s_means.npy')
        # conv1s_std = np.load(args.extract_path + '/conv1s_std.npy')
        with open(args.extract_path + '/inputs.json', 'r') as f:
            inputs = json.load(f)
    else:
        extract_paths = args.extract_path
        print('\nProcessing data in %s.' % extract_paths)

        save_path = pathlib.Path(extract_paths + '/epoch_%d' % 0)

        if not save_path.exists():
            # pdb.set_trace()
            raise FileExistsError(str(save_path))

        print('\rReading: ' + str(save_path), end='')
        # pdb.set_trace()
        input_uids = []

        for name in ['train', 'valid']:
            sets_files = list(save_path.glob('vox1_%s.*.bin' % name))
            uids = []
            for f in sets_files:
                with open(str(f), 'rb') as f:
                    sets = pickle.load(f)
                    for (uid, orig, conv1, bn1, relu1, grad) in sets:
                        uids.append(uid)
                        # in_mean += np.mean(orig, axis=0)
                        # num_utt += 1
            input_uids.append(uids)

        # inputs.append(input_uids)

        # inputs: [train/valid, 161]
        with open(args.extract_path + '/inputs.json', 'w') as f:
            json.dump(input_uids, f)

    # input_uids [train/valid, uids]
    #
    if os.path.exists(args.extract_path + '/inputs.npy'):
        all_data = np.load(args.extract_path + '/inputs.npy')
    else:
        feat_scp = os.path.join(args.train_dir, 'feats.scp')

        if not os.path.exists(feat_scp):
            raise FileExistsError(feat_scp)

        uid2feat_dict = {}
        with open(feat_scp, 'r') as u:
            all_cls = u.readlines()
            for line in all_cls:
                uid_feat = line.split()
                u = uid_feat[0]
                if u not in uid2feat_dict.keys():
                    uid2feat_dict[u] = uid_feat[1]

        all_data = []
        for s in subsets:
            aug_sets = []
            for i in range(len(input_uids)):
                train_valid = []
                for u in input_uids[i]:
                    uid = u[0] if s == 'orignal' else '-'.join((u[0], s))
                    feats = file_loader(uid2feat_dict[uid])
                    train_valid.append(np.mean(feats, axis=0))
                aug_sets.append(train_valid)
            all_data.append(aug_sets)

        all_data = np.array(all_data)
        np.save(args.extract_path + '/inputs.npy', all_data)
        print('Saving inputs in %s' % args.extract_path)

    # all_data [5, 2, 120, 161]
    # pdb.set_trace()
    # plotting filters distributions
    plt.figure(figsize=(10, 8))
    plt.title('SuResCNN 10', fontsize=25)
    plt.xlabel('Frequency', fontsize=18)
    plt.ylabel('Power Energy', fontsize=18)

    x = np.arange(161) * 8000 / 161  # [0-8000]
    y = np.sum(all_data, axis=2)  # [5,2,162]
    y = np.mean(y, axis=1)

    y1 = y[0]
    y2 = np.mean(y[1:], axis=0)

    # pdb.set_trace()
    # max_x = np.max(x)
    # min_x = np.min(x)
    # max_y = np.max(y)
    # min_y = np.min(y)
    # plt.xlim(min_x - 0.15 * np.abs(max_x), max_x + 0.15 * np.abs(max_x))
    # plt.ylim(min_y - 0.15 * np.abs(max_y), max_y + 0.15 * np.abs(max_y))
    # pdb.set_trace()
    # print(y.shape)
    y_shape = y.shape  # 5, 161

    lines = []
    al_data_label = subsets

    i = 0
    # for j in range(y_shape[0]):  # aug and kaldi

    # y2 = (y2 - np.mean(y2)) / (np.std(y2) + 2e-12)

    f = interpolate.interp1d(x, y1)
    xnew = np.arange(np.min(x), np.max(x), 500)
    ynew = f(xnew)

    plt.plot(xnew, ynew, color=cValue_1[i])

    f = interpolate.interp1d(x, y2)
    xnew = np.arange(np.min(x), np.max(x), 500)
    ynew = f(xnew)

    plt.plot(xnew, ynew, color=cValue_1[i + 1])
    # i += 1
    # lines.append(l)

    plt.legend(['Original data', 'Augmentation data'], loc='upper right', fontsize=18)
    plt.savefig(args.extract_path + "/inputs_1.png")
    plt.show()

    # plt.figure(figsize=(10, 8))
    # plt.title('Distribution of Data')
    # plt.xlabel('Frequency')
    # plt.ylabel('Power Energy')
    # # plt.xlim(min_x - 0.15 * np.abs(max_x), max_x + 0.15 * np.abs(max_x))
    # # plt.ylim(min_y - 0.15 * np.abs(max_y), max_y + 0.15 * np.abs(max_y))
    # for h in range(y_shape[0]):  # aug and kaldi
    #     y2 = y[h]
    #     # y2 = (y2 - np.mean(y2)) / (np.std(y2) + 2e-12)
    #
    #     f = interpolate.interp1d(x, y2)
    #     xnew = np.arange(np.min(x), np.max(x), 500)
    #     ynew = f(xnew)
    #
    #     l = plt.plot(xnew, ynew, color=cValue_1[i])
    #     i += 1
    #     lines.append(l)
    #
    # plt.legend(al_data_label[2:], loc='upper right', fontsize=15)
    # plt.savefig(args.extract_path + "/inputs_2.png")
    # plt.show()

if __name__ == '__main__':
    main()
