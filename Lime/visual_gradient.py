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
import os
import pdb
import pickle
import random
import torch

import numpy as np
import pathlib
import matplotlib.pyplot as plt
from kaldi_io import read_mat

from Process_Data import constants as c
from matplotlib import animation
from scipy import interpolate
from torch import nn
import torchvision.transforms as transforms
import json
from Lime import cValue_1, marker

from Process_Data.KaldiDataset import ScriptTrainDataset, ScriptValidDataset
from Process_Data.audio_processing import concateinputfromMFB, to2tensor

parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Model options
parser.add_argument('--extract-path', default='Lime/LoResNet10/center_dp0.00',
                    help='folder to output model checkpoints')
# Training options
parser.add_argument('--embedding-size', type=int, default=1024, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--seed', type=int, default=123456, metavar='S',
                    help='random seed (default: 0)')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

def main():

    # subsets = ['orignal', 'babble', 'noise', 'music', 'reverb']

    # load selected input uids
    dir_path = pathlib.Path(args.extract_path)

    # inputs [train/valid/test]
    if os.path.exists(args.extract_path + '/inputs.train.npy'):
        train_data = np.load(args.extract_path + '/inputs.train.npy')
        valid_data = np.load(args.extract_path + '/inputs.valid.npy')
        test_data = np.load(args.extract_path + '/inputs.test.npy')

    else:
        train_lst = list(dir_path.glob('*train*bin'))
        valid_lst = list(dir_path.glob('*valid*bin'))
        test_lst = list(dir_path.glob('*test*bin'))

        train_data = np.zeros(2, 161) # [data/grad]
        num_utt = 0
        for t in train_lst:
            p = str(t)
            with open(p, 'rb') as f:
                sets = pickle.load(f)
                for (data, grad) in sets:
                    train_data[1] += np.mean(np.abs(grad), axis=0)
                    train_data[0] += np.mean(data, axis=0)
                    num_utt += 1
        train_data = train_data / num_utt

        valid_data = np.zeros(2, 161)  # [data/grad]
        num_utt = 0
        for t in valid_lst:
            p = str(t)
            with open(p, 'rb') as f:
                sets = pickle.load(f)
                for (data, grad) in sets:
                    valid_data[1] += np.mean(np.abs(grad), axis=0)
                    valid_data[0] += np.mean(data, axis=0)
                    num_utt += 1
        valid_data = valid_data / num_utt

        test_data = np.zeros(2, 2, 161)  # [data/grad, utt_a, utt_b]
        num_utt = 0
        for t in test_lst:
            p = str(t)
            with open(p, 'rb') as f:
                sets = pickle.load(f)
                for (label, grad_a, grad_b, data_a, data_b) in sets:
                    test_data[0][0] += np.mean(data_a, axis=0)
                    test_data[0][1] += np.mean(data_b, axis=0)

                    test_data[1][0] += np.mean(np.abs(grad_a), axis=0)
                    test_data[1][1] += np.mean(np.abs(grad_b), axis=0)

                    num_utt += 1

        test_data = test_data / num_utt

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
    train_set_input = train_data[0]
    valid_set_input = valid_data[0]
    test_a_set_input = test_data[0][0]
    test_b_set_input = test_data[0][1]

    train_set_grad = train_data[1]
    valid_set_grad = valid_data[1]
    test_a_set_grad = test_data[1][0]
    test_b_set_grad = test_data[1][1]

    x = np.arange(161) * 8000 / 161  # [0-8000]
    # y = np.sum(all_data, axis=2)  # [5, 2, 162]


    y1 = y[0] # original data
    y2 = np.mean(y[1:], axis=0) # augmented

    y_shape = y.shape  # 5, 161
    # 插值平滑 ？？？
    for s in train_set_grad,valid_set_grad,test_a_set_grad,test_b_set_grad :
        f = interpolate.interp1d(x, s)
        xnew = np.arange(np.min(x), np.max(x), 500)
        ynew = f(xnew)
        plt.plot(xnew, ynew)

    plt.legend(['Train', 'Valid', 'Test_a', 'Test_b'], loc='upper right', fontsize=18)
    plt.savefig(args.extract_path + "/grads.png")
    plt.show()


if __name__ == '__main__':
    main()
