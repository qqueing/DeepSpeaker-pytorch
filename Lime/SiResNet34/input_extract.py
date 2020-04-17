#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: input_compare.py
@Time: 2020/3/25 5:30 PM
@Overview:
"""
import argparse
import json
import os
import pathlib
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from kaldi_io import read_mat
from scipy import interpolate

parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Model options
parser.add_argument('--train-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_aug_fb64/dev_no_sil',
                    help='path to dataset')
parser.add_argument('--test-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_aug_spect/test',
                    help='path to voxceleb1 test dataset')
parser.add_argument('--sitw-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/sitw_spect',
                    help='path to voxceleb1 test dataset')

parser.add_argument('--check-path', default='Data/checkpoint/SuResCNN10/spect/aug',
                    help='folder to output model checkpoints')
parser.add_argument('--extract-path', default='Lime/SiResNet34',
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

cValue_1 = ['#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000', '#95d0fc', '#029386', '#f97306',
            '#96f97b', '#c20078', '#ffff14', '#75bbfd', '#929591', '#89fe05', '#bf77f6', '#9a0eea', '#033500',
            '#06c2ac', '#c79fef', '#00035b', '#d1b26f', '#00ffff', '#13eac9', '#06470c', '#ae7181', '#35063e',
            '#01ff07', '#650021',
            '#6e750e', '#ff796c', '#e6daa6', '#0504aa', '#001146', '#cea2fd', '#000000', '#ff028d', '#ad8150',
            '#c7fdb5', '#ffb07c', '#677a04', '#cb416b', '#8e82fe', '#53fca1', '#aaff32', '#380282', '#ceb301',
            '#ffd1df', '#cf6275', '#0165fc', '#0cff0c', '#c04e01', '#04d8b2', '#01153e', '#3f9b0b', '#d0fefe',
            '#840000', '#be03fd', '#c0fb2d', '#a2cffe', '#dbb40c', '#8fff9f', '#580f41', '#4b006e', '#8f1402',
            '#014d4e', '#610023', '#aaa662', '#137e6d', '#7af9ab', '#02ab2e', '#9aae07', '#8eab12', '#b9a281',
            '#341c02', '#36013f', '#c1f80a', '#fe01b1', '#fdaa48', '#9ffeb0', '#b0ff9d', '#e2ca76', '#c65102',
            '#a9f971', '#a57e52', '#80f9ad', '#6b8ba4', '#4b5d16', '#363737', '#d5b60a', '#fac205', '#516572',
            '#90e4c1', '#a83c09', '#040273', '#ffcfdc', '#0485d1', '#ff474c', '#d2bd0a', '#bf9005', '#ffff84',
            '#8c000f', '#ed0dd9', '#0b4008', '#607c8e', '#5b7c99', '#b790d4', '#047495', '#d648d7', '#a5a502',
            '#d8dcd6', '#5ca904', '#fffe7a', '#380835', '#5a7d9a', '#658b38', '#98eff9', '#ffffff', '#789b73',
            '#87ae73', '#a03623', '#b04e0f', '#7f2b0a', '#ffffc2', '#fc5a50', '#03719c', '#40a368', '#960056',
            '#fd3c06', '#703be7', '#020035', '#d6b4fc', '#c0737a', '#2c6fbb', '#cdfd02', '#b0dd16', '#601ef9',
            '#5e819d', '#6c3461', '#acbf69', '#5170d7', '#f10c45', '#ff000d', '#069af3', '#5729ce', '#045c5a',
            '#0652ff', '#ffffe4', '#b1d1fc', '#80013f', '#74a662', '#76cd26', '#7ef4cc', '#bc13fe', '#1e488f',
            '#d46a7e', '#6f7632', '#0a888a', '#632de9', '#34013f', '#856798', '#154406', '#a2a415', '#ffa756',
            '#0b8b87', '#af884a', '#06b48b', '#10a674']
marker = ['o', 'x']


def main():
    # conv1s = np.array([]).reshape((0, 64, 5, 5))
    # grads = np.array([]).reshape((0, 2, 161))
    # model_set = ['kaldi_5wd', 'aug']
    subsets = ['orignal', 'babble', 'noise', 'music', 'reverb']
    file_loader = read_mat

    #
    if os.path.exists(args.extract_path + '/inputs.json'):
        # conv1s_means = np.load(args.extract_path + '/conv1s_means.npy')
        # conv1s_std = np.load(args.extract_path + '/conv1s_std.npy')
        with open(args.extract_path + '/inputs.json', 'r') as f:
            input_uids = json.load(f)
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
    plt.title('SiResNet 34', fontsize=25)
    plt.xlabel('Frequency', fontsize=18)
    plt.ylabel('Power Energy', fontsize=18)

    mel_high = 2595 * np.log10(1 + 8000 / 700)
    mel_cen = [mel_high / 65 * i for i in range(1, 65)]
    mel_cen = np.array(mel_cen)

    x = 700 * (10 ** (mel_cen / 2595) - 1)
    y = np.sum(all_data, axis=2)  # [5,2,162]
    y = np.mean(y, axis=1)

    y1 = y[0][1:]
    y2 = np.mean(y[1:], axis=0)[1:]

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
