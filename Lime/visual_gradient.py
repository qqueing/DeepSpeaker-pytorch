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
import pathlib
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from python_speech_features import mel2hz
from scipy import interpolate

parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Model options
parser.add_argument('--extract-path',
                    help='folder to output model checkpoints')
# Training options
parser.add_argument('--feat-dim', type=int, default=161, metavar='ES',
                    help='Dimensionality of the features')

parser.add_argument('--acoustic-feature', choices=['fbank', 'spectrogram', 'mfcc'], default='spectrogram',
                    help='choose the acoustic features type.')

parser.add_argument('--seed', type=int, default=123456, metavar='S',
                    help='random seed (default: 0)')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

def main():

    # subsets = ['orignal', 'babble', 'noise', 'music', 'reverb']

    # load selected input uids
    dir_path = pathlib.Path(args.extract_path)
    print('Path is %s' % str(dir_path))

    # inputs [train/valid/test]
    if os.path.exists(args.extract_path + '/inputs.train.npy'):
        train_data = np.load(args.extract_path + '/inputs.train.npy')
        valid_data = np.load(args.extract_path + '/inputs.valid.npy')
        test_data = np.load(args.extract_path + '/inputs.test.npy')
        veri_data = np.load(args.extract_path + '/inputs.veri.npy')

    else:
        train_lst = list(dir_path.glob('*train*bin'))
        veri_lst = list(dir_path.glob('*ver*bin'))
        valid_lst = list(dir_path.glob('*valid*bin'))
        test_lst = list(dir_path.glob('*test*bin'))

        print('Train set extracting:')
        train_data = np.zeros((3, args.feat_dim))  # [data/grad]
        num_utt = 0
        for t in train_lst:
            p = str(t)
            with open(p, 'rb') as f:
                sets = pickle.load(f)
                for (data, grad) in sets:
                    train_data[1] += np.sum(np.abs(grad), axis=0)
                    this_weight = np.var(grad, axis=0)
                    train_data[2] += this_weight  # / this_weight.sum()
                    # train_data[1] += np.mean(grad, axis=0)
                    train_data[0] += np.mean(data, axis=0)
                    num_utt += 1
        train_data = train_data / num_utt

        print('Valid set extracting:')
        valid_data = np.zeros((3, args.feat_dim))  # [data/grad]
        num_utt = 0
        for t in valid_lst:
            p = str(t)
            with open(p, 'rb') as f:
                sets = pickle.load(f)
                for (data, grad) in sets:
                    valid_data[1] += np.sum(np.abs(grad), axis=0)
                    this_weight = np.var(grad, axis=0)
                    valid_data[2] += this_weight  # / this_weight.sum()
                    # valid_data[1] += np.mean(grad, axis=0)
                    valid_data[0] += np.mean(data, axis=0)
                    num_utt += 1
        valid_data = valid_data / num_utt

        print('Train verification set extracting:')
        veri_data = np.zeros((3, 2, args.feat_dim))  # [data/grad, utt_a, utt_b]
        num_utt = 0
        for t in veri_lst:
            p = str(t)
            with open(p, 'rb') as f:
                sets = pickle.load(f)
                for (label, grad_a, grad_b, data_a, data_b) in sets:
                    veri_data[0][0] += np.mean(data_a, axis=0)
                    veri_data[0][1] += np.mean(data_b, axis=0)

                    veri_data[1][0] += np.sum(np.abs(grad_a), axis=0)
                    veri_data[1][1] += np.sum(np.abs(grad_b), axis=0)

                    this_weight_a = np.var(grad_a, axis=0)
                    veri_data[2][0] += this_weight_a  # / this_weight_a.sum()

                    this_weight_b = np.var(grad_b, axis=0)
                    veri_data[2][1] += this_weight_b  # / this_weight_b.sum()

                    num_utt += 1

        veri_data = veri_data / num_utt

        print('Test set extracting:')
        test_data = np.zeros((3, 2, args.feat_dim))  # [data/grad, utt_a, utt_b]
        num_utt = 0
        for t in test_lst:
            p = str(t)
            with open(p, 'rb') as f:
                sets = pickle.load(f)
                for (label, grad_a, grad_b, data_a, data_b) in sets:
                    test_data[0][0] += np.mean(data_a, axis=0)
                    test_data[0][1] += np.mean(data_b, axis=0)

                    test_data[1][0] += np.sum(np.abs(grad_a), axis=0)
                    test_data[1][1] += np.sum(np.abs(grad_b), axis=0)

                    this_weight_a = np.var(grad_a, axis=0)
                    test_data[2][0] += this_weight_a  # / this_weight_a.sum()

                    this_weight_b = np.var(grad_b, axis=0)
                    test_data[2][1] += this_weight_b  #/ this_weight_b.sum()

                    num_utt += 1

        test_data = test_data / num_utt

        print('Saving inputs in %s' % args.extract_path)

        train_data = np.array(train_data)
        valid_data = np.array(valid_data)
        test_data = np.array(test_data)

        np.save(args.extract_path + '/inputs.train.npy', train_data)
        np.save(args.extract_path + '/inputs.valid.npy', valid_data)
        np.save(args.extract_path + '/inputs.veri.npy', veri_data)
        np.save(args.extract_path + '/inputs.test.npy', test_data)

    # all_data [5, 2, 120, 161]
    # plotting filters distributions

    # train_data [numofutt, feats[N, 161]]
    train_set_input = train_data[0]
    valid_set_input = valid_data[0]
    test_a_set_input = test_data[0][0]
    test_b_set_input = test_data[0][1]

    train_set_grad = train_data[1]
    valid_set_grad = valid_data[1]

    veri_set_grad = veri_data[1][0] + veri_data[1][1]
    test_set_grad = test_data[1][0] + test_data[1][1]

    x = np.arange(args.feat_dim) * 8000 / (args.feat_dim - 1)  # [0-8000]
    if args.acoustic_feature == 'fbank':
        m = np.linspace(0, 2840.0230467083188, args.feat_dim)
        x = mel2hz(m)

    # y = np.sum(all_data, axis=2)  # [5, 2, 162]

    pdf = PdfPages(args.extract_path + '/grad.veri.pdf')
    plt.rc('font', family='Times New Roman')

    plt.figure(figsize=(12, 9))
    # plt.title('Gradient Distributions', fontsize=22)
    plt.xlabel('Frequency (Hz)', fontsize=24)
    plt.xticks(fontsize=22)
    plt.ylabel('Weight', fontsize=24)
    plt.yticks(fontsize=22)

    m = np.arange(0, 2840.0230467083188)
    m = 700 * (10 ** (m / 2595.0) - 1)
    n = np.array([m[i] - m[i - 1] for i in range(1, len(m))])
    n = 1 / n

    f = interpolate.interp1d(m[1:], n)
    xnew = np.arange(np.min(m[1:]), np.max(m[1:]), (np.max(m[1:]) - np.min(m[1:])) / 161)
    ynew = f(xnew)
    ynew = ynew / ynew.sum()
    plt.plot(xnew, ynew)
    # print(np.sum(ynew))

    for s in train_set_grad, valid_set_grad, veri_set_grad, test_set_grad:
        # for s in test_a_set_grad, test_b_set_grad:
        f = interpolate.interp1d(x, s)
        xnew = np.linspace(np.min(x), np.max(x), 161)
        ynew = f(xnew)
        # ynew = ynew - ynew.min()
        ynew = ynew / ynew.sum()
        plt.plot(xnew, ynew)
        # pdb.set_trace
    # if not os.path.exists(args.extract_path + '/grad.npy'):
    ynew = veri_set_grad
    ynew = ynew / ynew.sum()

    np.save(args.extract_path + '/grad.veri.npy', ynew)

    # plt.legend(['Mel-scale', 'Train', 'Valid', 'Test_a', 'Test_b'], loc='upper right', fontsize=18)
    plt.legend(['Mel-scale', 'Train Set', 'Valid Set', 'train Verify Set', 'Test Set'], loc='upper right', fontsize=24)
    pdf.savefig()
    pdf.close()

    # plt.savefig(args.extract_path + "/grads.png")
    # plt.show()

    plt.figure(figsize=(8, 6))
    plt.title('Data distributions', fontsize=22)
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.ylabel('Log Power Energy (CMVN)', fontsize=16)
    # 插值平滑 ？？？
    for s in train_set_input, valid_set_input, test_a_set_input, test_b_set_input:
        # for s in test_a_set_grad, test_b_set_grad:
        f = interpolate.interp1d(x, s)
        xnew = np.linspace(np.min(x), np.max(x), 161)
        ynew = f(xnew)
        plt.plot(xnew, ynew)

    plt.legend(['Train', 'Valid', 'Test_a', 'Test_b'], loc='upper right', fontsize=16)
    plt.savefig(args.extract_path + "/inputs.png")
    plt.show()
    print('Completed!\n')

if __name__ == '__main__':
    main()
