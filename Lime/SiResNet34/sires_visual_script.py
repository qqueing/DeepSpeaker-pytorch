#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: sires_visual_script.py
@Time: 2020/3/23 12:59 AM
@Overview:
"""

import os
import argparse
import pdb
import pickle
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from matplotlib import animation

parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Model options
parser.add_argument('--train-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/dev_no_sil',
                    help='path to dataset')
parser.add_argument('--test-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_fb64/test_no_sil',
                    help='path to voxceleb1 test dataset')
parser.add_argument('--sitw-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/sitw_spect',
                    help='path to voxceleb1 test dataset')

parser.add_argument('--check-path', default='Data/checkpoint/SuResCNN10/spect/kaldi_5wd',
                    help='folder to output model checkpoints')
parser.add_argument('--extract-path', default='Data/extract/SiResNet34/soft',
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

cValue_1 = ['purple', 'green', 'blue', 'pink', 'brown', 'red', 'teal', 'orange', 'magenta', 'yellow', 'grey',
            'violet', 'turquoise', 'lavender', 'tan', 'cyan', 'aqua', 'maroon', 'olive', 'salmon', 'beige', 'lilac',
            'black', 'peach', 'lime', 'indigo', 'mustard', 'rose', 'aquamarine', 'navy', 'gold', 'plum', 'burgundy',
            'khaki', 'taupe', 'chartreuse', 'mint', 'sand', 'puce', 'seafoam', 'goldenrod', 'slate', 'rust',
            'cerulean', 'ochre', 'crimson', 'fuchsia', 'puke', 'eggplant', 'white', 'sage', 'brick', 'cream',
            'coral', 'greenish', 'grape', 'azure', 'wine', 'cobalt', 'pinkish', 'vomit', 'moss', 'grass',
            'chocolate', 'cornflower', 'charcoal', 'pumpkin', 'tangerine', 'raspberry', 'orchid', 'sky']
marker = ['o', 'x']

def main():
    model_set = ['kaldi', 'aug']
    epochs = np.arange(0, 31)

    if os.path.exists(args.extract_path + '/conv1s_means.npy'):
        conv1s_means = np.load(args.extract_path + '/conv1s_means.npy')
        conv1s_std = np.load(args.extract_path + '/conv1s_std.npy')
        input_grads = np.load(args.extract_path + '/input_grads.npy')
    else:
        conv1s_means = []
        conv1s_std = []
        input_grads = []

        for model in model_set:
            extract_paths = os.path.join(args.extract_path, model)
            conv1s = np.array([]).reshape((0, 16, 3, 3))
            grads = np.array([]).reshape((0, 2, 64))
            print('\nProcessing data in %s.' % extract_paths)

            for i in epochs:
                save_path = pathlib.Path(extract_paths + '/epoch_%d' % i)
                print('\rReading: ' + str(save_path), end='')
                if not save_path.exists():
                    continue
                grads_abs = np.array([]).reshape((0, 64))

                for name in ['train', 'valid']:
                    sets_files = list(save_path.glob('vox1_%s.*.bin' % name))
                    grad_abs = np.zeros((64))
                    num_utt = 0
                    for f in sets_files:
                        with open(str(f), 'rb') as f:
                            sets = pickle.load(f)
                            for (uid, orig, conv1, bn1, relu1, grad) in sets:
                                # pdb.set_trace()
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

        conv1s_means = np.array(conv1s_means)  # [[2,21,16]; [2,30,16]]
        conv1s_std = np.array(conv1s_std)  # 2,21,16
        input_grads = np.array(input_grads)  # 2,21,64

        np.save(args.extract_path + '/conv1s_means.npy', conv1s_means)
        np.save(args.extract_path + '/conv1s_std.npy', conv1s_std)
        np.save(args.extract_path + '/input_grads.npy', input_grads)


    # plotting filters distributions
    plt.rc('font', family='Times New Roman')
    fig = plt.figure(figsize=(8, 8))
    plt.title('Convergence of 16 Filters')

    # x(mean): [2, 2, 30, 16] [model, aug/kaldi, epoch, filters]
    # y(std): ~
    max_x = np.max(conv1s_means)
    min_x = np.min(conv1s_means)
    max_y = np.max(conv1s_std)
    min_y = np.min(conv1s_std)

    plt.xlim(min_x - 0.15 * np.abs(max_x), max_x + 0.15 * np.abs(max_x))
    plt.ylim(min_y - 0.15 * np.abs(max_y), max_y + 0.15 * np.abs(max_y))
    plt.xlabel('Means of Abs')
    plt.ylabel('Std')

    # fig, ax = plt.subplots()
    means_shape = conv1s_means.shape  # 2,21,16
    set_dots = []
    text_e = plt.text(max_x, max_y, 'Epoch 0')

    for i in range(means_shape[0]):  # aug, kaldi
        dots = []
        for j in range(means_shape[2]):
            dot_x = conv1s_means[i][0][j]
            dot_y = conv1s_std[i][0][i]

            dot, = plt.plot(dot_x, dot_y, color=cValue_1[j], marker=marker[i])
            text_p = plt.text(dot_x, dot_y, '%d' % j)
            dots.append([dot, text_p])
        set_dots.append(dots)

    plt.legend([set_dots[0][0][0], set_dots[1][0][0]], ['aug', 'kaldi'], loc='lower right', scatterpoints=1)

    def gen_dot():
        for i in range(means_shape[1]):
            text_e.set_text('Epoch %2s' % str(i))
            newdot = [conv1s_means[:, i], conv1s_std[:, i]]
            yield newdot

    def update_dot(newd):
        # pdb.set_trace()
        for i in range(means_shape[0]):
            dots = set_dots[i]
            for j in range(means_shape[2]):
                dot, text_p = dots[j]

                dot_x = newd[0][i][j]
                dot_y = newd[1][i][j]

                dot.set_data(dot_x, dot_y)
                text_p.set_position((dot_x, dot_y))

        return set_dots

    ani = animation.FuncAnimation(fig, update_dot, frames=gen_dot, interval=800)
    ani.save(args.extract_path + "/conv1s.gif", writer='pillow', fps=2)
    print('Saving %s' % args.extract_path + "/conv1s.gif")

    fig = plt.figure(figsize=(10, 8))
    plt.title('Filting over 8000Hz')
    plt.xlabel('Frequency')
    plt.ylabel('Weight')

    mel_high = 2595 * np.log10(1 + 8000 / 700)
    mel_cen = [mel_high / 65 * i for i in range(1, 65)]
    mel_cen = np.array(mel_cen)

    x = 700 * (10 ** (mel_cen / 2595) - 1)
    y = np.nan_to_num(input_grads)
    # pdb.set_trace()
    max_x = np.max(x)
    min_x = np.min(x)
    max_y = np.max(y)
    min_y = np.min(y)
    plt.xlim(min_x - 0.15 * np.abs(max_x), max_x + 0.15 * np.abs(max_x))
    plt.ylim(min_y - 0.15 * np.abs(max_y), max_y + 0.15 * np.abs(max_y))
    # pdb.set_trace()
    # print(y.shape)
    text_e = plt.text(min_x, max_y, 'Epoch 0')
    y_shape = y.shape  # 2, 21, 2, 64
    set_dots = []

    for j in range(y_shape[0]):  # aug and kaldi
        dots = []
        for h in range(y_shape[2]):  # train and valid
            dot, = plt.plot(x, y[j][0][h], marker=marker[j], color=cValue_1[j + h * 4])
            dots.append(dot)

        set_dots.append(dots)

    plt.legend([set_dots[0][0], set_dots[0][1], set_dots[1][0], set_dots[1][1]],
               ['kaldi_Train', 'kaldi_Valid', 'aug_Train', 'aug_Valid'], loc='upper right')

    def gen_line():
        for i in range(1, y_shape[1]):
            newdot = [x, y[:, i]]  # 2,2,64
            text_e.set_text('Epoch %2s' % str(i))

            yield newdot

    def update_line(newd):
        for i in range(y_shape[0]):
            dots = set_dots[i]
            for j in range(y_shape[2]):
                dots[j].set_data(newd[0], newd[1][i][j])

        return set_dots

    ani = animation.FuncAnimation(fig, update_line, frames=gen_line, interval=800)
    ani.save(args.extract_path + "/grads.gif", writer='pillow', fps=2)
    # plt.show()


if __name__ == '__main__':
    main()
