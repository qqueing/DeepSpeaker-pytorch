#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: output_visual_script.py
@Time: 2020/3/21 10:43 PM
@Overview:
"""
import argparse
import os
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Model options
parser.add_argument('--train-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_aug_spect/dev_org',
                    help='path to dataset')
parser.add_argument('--test-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_aug_spect/test',
                    help='path to voxceleb1 test dataset')
parser.add_argument('--sitw-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/sitw_spect',
                    help='path to voxceleb1 test dataset')

parser.add_argument('--check-path', default='Data/checkpoint/SuResCNN10/spect/aug',
                    help='folder to output model checkpoints')
parser.add_argument('--extract-path', default='Lime/SuResCNN10/data',
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

args = parser.parse_args()
cValue_1 = ['#15b01a', '#ff796c', '#7e1e9c', '#ff81c0', '#96f97b', '#e50000', '#ffff14', '#653700', '#95d0fc',
            '#029386', '#f97306',
            '#c20078', '#75bbfd', '#929591', '#89fe05', '#bf77f6', '#9a0eea', '#033500', '#06c2ac', '#c79fef',
            '#00035b', '#d1b26f', '#00ffff', '#13eac9', '#06470c', '#ae7181', '#35063e', '#01ff07', '#650021',
            '#6e750e', '#e6daa6', '#0504aa', '#001146', '#cea2fd', '#000000', '#ff028d', '#ad8150', '#c7fdb5',
            '#ffb07c', '#677a04', '#cb416b', '#8e82fe', '#53fca1', '#aaff32', '#380282', '#ceb301', '#ffd1df',
            '#cf6275', '#0165fc', '#0cff0c', '#c04e01', '#04d8b2', '#01153e', '#3f9b0b', '#d0fefe', '#840000',
            '#be03fd', '#c0fb2d', '#a2cffe', '#dbb40c', '#8fff9f', '#580f41', '#4b006e', '#8f1402', '#014d4e',
            '#610023', '#aaa662', '#137e6d', '#7af9ab', '#02ab2e', '#9aae07', '#8eab12', '#b9a281', '#341c02',
            '#36013f', '#c1f80a', '#fe01b1', '#fdaa48', '#9ffeb0', '#b0ff9d', '#e2ca76', '#c65102', '#a9f971',
            '#a57e52', '#80f9ad', '#6b8ba4', '#4b5d16', '#363737', '#d5b60a', '#fac205', '#516572', '#0343df',
            '#90e4c1', '#a83c09', '#040273', '#ffcfdc', '#0485d1', '#ff474c', '#d2bd0a', '#bf9005', '#ffff84',
            '#8c000f', '#ed0dd9', '#0b4008', '#607c8e', '#5b7c99', '#b790d4', '#047495', '#d648d7', '#a5a502',
            '#d8dcd6', '#5ca904', '#fffe7a', '#380835', '#5a7d9a', '#658b38', '#98eff9', '#ffffff', '#789b73',
            '#87ae73', '#a03623', '#b04e0f', '#7f2b0a', '#ffffc2', '#fc5a50', '#03719c', '#40a368', '#960056',
            '#fd3c06', '#703be7', '#020035', '#d6b4fc', '#c0737a', '#2c6fbb', '#cdfd02', '#b0dd16', '#601ef9',
            '#5e819d', '#6c3461', '#acbf69', '#5170d7', '#f10c45', '#ff000d', '#069af3', '#5729ce', '#045c5a',
            '#0652ff', '#ffffe4', '#b1d1fc', '#80013f', '#74a662', '#76cd26', '#7ef4cc', '#bc13fe', '#1e488f',
            '#d46a7e', '#6f7632', '#0a888a', '#632de9', '#34013f', '#856798', '#154406', '#a2a415', '#ffa756',
            '#0b8b87', '#af884a', '#06b48b', '#10a674', '#a2bffe', '#769958', '#5cac2d', '#cb0162', '#980002',
            '#88b378', '#02d8e9', '#ca6641', '#caa0ff', '#a9561e', '#373e02', '#c9ff27', '#be0119', '#82a67d',
            '#3d1c02', '#5d06e9', '#6a79f7', '#ffb7ce', '#343837', '#0a481e', '#e17701', '#696112', '#8b2e16',
            '#6a6e09', '#ff9408', '#fe7b7c', '#12e193', '#b00149', '#887191', '#f7879a', '#fe019a', '#030aa7',
            '#be6400', '#9a0200', '#fd411e', '#cdc50a']
marker = ['.', '*']

def main():
    # conv1s = np.array([]).reshape((0, 64, 5, 5))
    # grads = np.array([]).reshape((0, 2, 161))
    model_set = ['kaldi_5wd', 'aug']
    epochs = np.arange(0, 31)

    if os.path.exists(args.extract_path + '/conv1s_means.npy'):
        conv1s_means = np.load(args.extract_path + '/conv1s_means.npy')
        conv1s_std = np.load(args.extract_path + '/conv1s_std.npy')
        input_grads = np.load(args.extract_path + '/input_grads.npy')
    else:
        conv1s_means = []
        conv1s_std = []
        input_grads = []

        for m in model_set:
            extract_paths = os.path.join(args.extract_path, m)
            conv1s = np.array([]).reshape((0, 64, 5, 5))
            grads = np.array([]).reshape((0, 2, 161))
            print('\nProcessing data in %s.' % extract_paths)

            for i in epochs:
                save_path = pathlib.Path(extract_paths + '/epoch_%d' % i)

                if not save_path.exists():
                    # pdb.set_trace()
                    print(str(save_path) + ' ERROR!')
                    continue

                print('\rReading: ' + str(save_path), end='')
                # pdb.set_trace()
                grads_abs = np.array([]).reshape((0, 161))

                for name in ['train', 'valid']:
                    sets_files = list(save_path.glob('vox1_%s.*.bin' % name))
                    grad_abs = np.zeros((161))
                    num_utt = 0
                    for f in sets_files:
                        with open(str(f), 'rb') as f:
                            sets = pickle.load(f)
                            for (uid, orig, conv1, bn1, relu1, grad) in sets:
                                grad_abs += np.mean(np.abs(grad), axis=0)
                                num_utt += 1
                    grads_abs = np.concatenate((grads_abs, grad_abs[np.newaxis, :] / num_utt), axis=0)

                grads_abs = grads_abs[np.newaxis, :]
                grads = np.concatenate((grads, grads_abs), axis=0)

                cs = list(save_path.glob('model.conv1.npy'))

                conv1_epoch = np.load(str(cs[0])).squeeze()
                conv1_epoch = conv1_epoch[np.newaxis, :]
                # pdb.set_trace()
                conv1s = np.concatenate((conv1s, conv1_epoch), axis=0)

            means = np.mean(np.abs(conv1s), axis=(2, 3))
            stds = np.std(conv1s, axis=(2, 3))

            conv1s_means.append(means)
            conv1s_std.append(stds)
            input_grads.append(grads)

        for x in conv1s_means, conv1s_std:
            while x[0].shape[0] < x[1].shape[0]:
                x[0] = np.concatenate((x[0], x[0][-1, :].reshape(1, x[0].shape[1])), axis=0)

        while input_grads[0].shape[0] < input_grads[1].shape[0]:
            input_grads[0] = np.concatenate(
                (input_grads[0], input_grads[0][-1, :, :].reshape(1, input_grads[0].shape[1], input_grads[0].shape[2])),
                axis=0)

        conv1s_means = np.array(conv1s_means)  # [[2,21,64]; [2,30,64]]
        conv1s_std = np.array(conv1s_std)  # 2,21,64
        input_grads = np.array(input_grads)  # 2,21,64

        np.save(args.extract_path + '/conv1s_means.npy', conv1s_means)
        np.save(args.extract_path + '/conv1s_std.npy', conv1s_std)
        np.save(args.extract_path + '/input_grads.npy', input_grads)

    # plotting filters distributions
    # fig = plt.figure(figsize=(10, 10))
    # plt.title('Convergence of 16 Filters')
    # # pdb.set_trace()
    #
    # max_x = max(np.max(conv1s_means[0][1]), np.max(conv1s_means[1][1]))
    # min_x = min(np.min(conv1s_means[0][1]), np.min(conv1s_means[1][1]))
    # max_y = max(np.max(conv1s_std[0][1]), np.max(conv1s_std[1][1]))
    # min_y = min(np.min(conv1s_std[0][1]), np.min(conv1s_std[1][1]))
    #
    # plt.xlim(min_x - 0.15 * np.abs(max_x), max_x + 0.15 * np.abs(max_x))
    # plt.ylim(min_y - 0.15 * np.abs(max_y), max_y + 0.15 * np.abs(max_y))
    # plt.xlabel('Means of Abs')
    # plt.ylabel('Std')

    # fig, ax = plt.subplots()
    # means_shape = conv1s_means.shape  # 2,21,16
    # set_dots = []
    # text_e = plt.text(max_x, max_y, 'Epoch 0')
    #
    # for i in range(means_shape[0]):  # aug, kaldi
    #     dots = []
    #     for j in range(means_shape[2]):
    #         dot_x = conv1s_means[i][0][j]
    #         dot_y = conv1s_std[i][0][i]
    #
    #         dot, = plt.plot(dot_x, dot_y, color=cValue_1[j], marker=marker[i])
    #         text_p = plt.text(dot_x, dot_y, '%d' % j)
    #         dots.append([dot, text_p])
    #     set_dots.append(dots)
    #
    # plt.legend([set_dots[0][0][0], set_dots[1][0][0]], ['aug', 'kaldi'], loc='lower right', scatterpoints=1)
    #
    # def gen_dot():
    #     for i in range(means_shape[1]):
    #         text_e.set_text('Epoch %2s' % str(i))
    #         newdot = [conv1s_means[:, i], conv1s_std[:, i]]
    #         yield newdot
    #
    # def update_dot(newd):
    #     # pdb.set_trace()
    #     for i in range(means_shape[0]):
    #         dots = set_dots[i]
    #         for j in range(means_shape[2]):
    #             dot, text_p = dots[j]
    #
    #             dot_x = newd[0][i][j]
    #             dot_y = newd[1][i][j]
    #
    #             dot.set_data(dot_x, dot_y)
    #             text_p.set_position((dot_x, dot_y))
    #
    #     return set_dots
    #
    # ani = animation.FuncAnimation(fig, update_dot, frames=gen_dot, interval=800)
    # ani.save(args.extract_path + "/conv1s.gif", writer='pillow', fps=2)
    # print('Saving %s' % args.extract_path + "/conv1s.gif")

    fig = plt.figure(figsize=(10, 8))
    plt.title('Filting over 8000Hz')
    plt.xlabel('Frequency')
    plt.ylabel('Weight')

    x = np.arange(161) * 8000 / 161  # [0-8000]
    y = np.nan_to_num(input_grads)  # 2, 21, 2, 161
    # pdb.set_trace()
    max_x = np.max(x)
    min_x = np.min(x)
    max_y = np.max(y[0][0][0] / y[0][0][0].sum())
    min_y = np.min(y)
    plt.xlim(min_x - 0.15 * np.abs(max_x), max_x + 0.15 * np.abs(max_x))
    plt.ylim(min_y, max_y + 0.15 * np.abs(max_y))
    # pdb.set_trace()
    # print(y.shape)
    text_e = plt.text(min_x, max_y, 'Epoch 0')
    y_shape = y.shape  # 2, 21, 2, 161
    set_dots = []

    for j in range(y_shape[0]):  # aug and kaldi
        dots = []
        for h in range(y_shape[2]):  # train and valid
            # dot, = plt.plot(x, y[j][0][h] / y[j][0][h].sum(), marker=marker[j], color=cValue_1[j + h * 4])
            # dot, = plt.plot(x, y[j][0][h] / y[j][0][h].sum(), marker=marker[j], alpha=0.8)
            dot, = plt.plot(x, y[j][0][h] / y[j][0][h].sum(), alpha=0.7)
            dots.append(dot)

        set_dots.append(dots)

    plt.legend([set_dots[0][0], set_dots[0][1], set_dots[1][0], set_dots[1][1]],
               ['kaldi_Train', 'kaldi_Valid', 'aug_Train', 'aug_Valid'], loc='upper right')

    def gen_line():
        for i in range(1, y_shape[1]):
            newdot = [x, y[:, i]]  # 2,2,161
            text_e.set_text('Epoch %2s' % str(i))

            yield newdot

    def update_line(newd):
        for i in range(y_shape[0]):
            dots = set_dots[i]
            for j in range(y_shape[2]):
                dots[j].set_data(newd[0], newd[1][i][j] / newd[1][i][j].sum())

        return set_dots

    ani = animation.FuncAnimation(fig, update_line, frames=gen_line, interval=800)
    ani.save(args.extract_path + "/grads.gif", writer='pillow', fps=2)


if __name__ == '__main__':
    main()
