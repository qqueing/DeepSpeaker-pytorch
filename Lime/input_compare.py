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
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from kaldi_io import read_mat
from scipy import interpolate

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
parser.add_argument('--extract-path', default='Lime/LoResNet10/center_dp0.00',
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

    subsets = ['orignal', 'babble', 'noise', 'music', 'reverb']
    file_loader = read_mat

    # load selected input uids
    if os.path.exists(args.extract_path + '/epoch_0/inputs.vox1_train.1.json'):
        # Lime/LoResNet10/data/epoch0/inputs.vox1_train.1.json
        with open(args.extract_path + '/epoch_0/inputs.vox1_train.1.json', 'r') as f:
            train_uids = json.load(f)

        with open(args.extract_path + '/epoch_0/inputs.vox1_valid.1.json', 'r') as f:
            valid_uids = json.load(f)

        with open(args.extract_path + '/epoch_0/inputs.vox1_test.1.json', 'r') as f:
            test_uids = json.load(f)
    else:
        raise FileNotFoundError('Utterance uids.')

    # input_uids [train/valid, uids]
    if os.path.exists(args.extract_path + '/inputs.train.npy'):
        train_data = np.load(args.extract_path + '/inputs.train.npy')
        valid_data = np.load(args.extract_path + '/inputs.valid.npy')
        test_data = np.load(args.extract_path + '/inputs.test.npy')

    else:
        feat_scp = os.path.join(args.train_dir, 'feats.scp')
        assert os.path.exists(feat_scp)

        uid2feat_dict = {}
        with open(feat_scp, 'r') as u:
            all_cls = u.readlines()
            for line in all_cls:
                uid_feat = line.split()
                u = uid_feat[0]
                if u not in uid2feat_dict.keys():
                    uid2feat_dict[u] = uid_feat[1]

        test_feat_scp = os.path.join(args.test_dir, 'feats.scp')
        assert os.path.exists(test_feat_scp)
        test_uid2feat_dict = {}
        with open(test_feat_scp, 'r') as u:
            all_cls = u.readlines()
            for line in all_cls:
                uid_feat = line.split()
                u = uid_feat[0]
                if u not in uid2feat_dict.keys():
                    test_uid2feat_dict[u] = uid_feat[1]

        train_data = []
        valid_data = []
        test_data = []

        for uid in train_uids:
            feats = file_loader(uid2feat_dict[uid])
            train_data.append(feats)

        for uid in valid_uids:
            feats = file_loader(uid2feat_dict[uid])
            valid_data.append(feats)

        for uid_a, uid_b in test_uids:
            feat_a = file_loader(test_uid2feat_dict[uid_a])
            feat_b = file_loader(test_uid2feat_dict[uid_b])
            test_data.append(feat_a, feat_b)

        print('Saving inputs in %s' % args.extract_path)

        train_data = np.array(train_data)
        valid_data = np.array(valid_data)
        test_data = np.array(test_data)

        np.save(args.extract_path + '/inputs.train.npy', train_data)
        np.save(args.extract_path + '/inputs.valid.npy', valid_data)
        np.save(args.extract_path + '/inputs.test.npy', test_data)

    # all_data [5, 2, 120, 161]
    # plotting filters distributions

    plt.figure(figsize=(10, 8))
    plt.title('Data distributions', fontsize=25)
    plt.xlabel('Frequency', fontsize=18)
    plt.ylabel('Log Power Energy (CMVN)', fontsize=18)

    # train_data [numofutt, feats[N, 161]]
    train_set_input = np.zeros(161)
    for u in train_data:
        train_set_input += np.mean(u, axis=0)
    train_set_input = train_set_input / len(train_data)

    valid_set_input = np.zeros(161)
    for u in valid_data:
        valid_set_input += np.mean(u, axis=0)
    valid_set_input = valid_set_input / len(valid_data)

    test_set_input = np.zeros(161)
    for a, b in test_data:
        test_set_input += np.mean(a, axis=0)
        test_set_input += np.mean(b, axis=0)

    test_set_input = test_set_input / len(test_data) / 2

    x = np.arange(161) * 8000 / 161  # [0-8000]
    # y = np.sum(all_data, axis=2)  # [5, 2, 162]

    y1 = y[0]  # original data
    y2 = np.mean(y[1:], axis=0)  # augmented

    y_shape = y.shape  # 5, 161
    # 插值平滑 ？？？
    f = interpolate.interp1d(x, y1)
    xnew = np.arange(np.min(x), np.max(x), 500)
    ynew = f(xnew)
    plt.plot(xnew, ynew)

    f = interpolate.interp1d(x, y2)
    xnew = np.arange(np.min(x), np.max(x), 500)
    ynew = f(xnew)
    plt.plot(xnew, ynew)

    plt.legend(['Original data', 'Augmentation data'], loc='upper right', fontsize=18)
    plt.savefig(args.extract_path + "/inputs_1.png")
    plt.show()


if __name__ == '__main__':
    main()
