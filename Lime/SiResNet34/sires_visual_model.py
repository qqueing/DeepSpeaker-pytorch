#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: sires_visual_model.py
@Time: 2020/3/24 2:53 PM
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

    if os.path.exists(args.extract_path + '/model_fc1.npy'):
        model_fc1 = np.load(args.extract_path + '/model_fc1.npy')
    else:
        model_fc1 = []

        for model in model_set:
            extract_paths = os.path.join(args.extract_path, model)
            fc1_weights = np.array([]).reshape((0, 128, 128))
            fc1_bias = np.array([]).reshape((0, 128))

            print('\nProcessing data in %s.' % extract_paths)

            for i in epochs:
                save_path = os.path.join(extract_paths, 'epoch_%d' % i)
                print('\rReading: ' + str(save_path), end='')
                if not os.path.exists(save_path):
                    continue

                fc_weight_path = os.path.join(save_path, 'model.fc1.weight.npy')
                fc_bias_path = os.path.join(save_path, 'model.fc1.bias.npy')

                weight = np.load(fc_weight_path)
                bias = np.load(fc_bias_path)

                fc1_weights = np.concatenate((fc1_weights, weight[np.newaxis, :]), axis=0)
                fc1_bias = np.concatenate((fc1_bias, bias[np.newaxis, :]), axis=0)

            weight_means = np.mean(np.abs(fc1_weights), axis=(1, 2))  # 21,
            weight_stds = np.std(fc1_weights, axis=(1, 2))  # 21,

            # this_weight = np.concatenate((weight_means[np.newaxis, :],
            #                            weight_stds[np.newaxis, :]), axis=0)  # shape: 2,21
            bias_means = np.mean(np.abs(fc1_bias), axis=1)  # 21,
            bias_stds = np.std(fc1_bias, axis=1)  # 21,
            # this_bias = np.concatenate((bias_means[np.newaxis, :],
            #                             bias_stds[np.newaxis, :]), axis=0)  # shape: 4,21

            this_fc1 = np.concatenate((weight_means[np.newaxis, :],
                                       weight_stds[np.newaxis, :],
                                       bias_means[np.newaxis, :],
                                       bias_stds[np.newaxis, :]), axis=0)  # shape: 4,21

            model_fc1.append(this_fc1)

        while model_fc1[0].shape[1] < model_fc1[1].shape[1]:
            model_fc1[0] = np.concatenate((model_fc1[0], model_fc1[0][:, -1].reshape(model_fc1[0].shape[0], 1)),
                                          axis=1)

        model_fc1 = np.array(model_fc1)  # [2, 4, 31]
        np.save(args.extract_path + '/model_fc1.npy', model_fc1)

    # plotting filters distributions
    fig = plt.figure(figsize=(10, 10))
    plt.title('Convergence of fc1')

    max_x = np.max(model_fc1)
    min_x = np.min(model_fc1)
    plt.xlim(min_x - 0.15 * np.abs(max_x), max_x + 0.15 * np.abs(max_x))
    plt.ylim(min_x - 0.15 * np.abs(max_x), max_x + 0.15 * np.abs(max_x))

    plt.xlabel('Means of Abs')
    plt.ylabel('Std')

    # fig, ax = plt.subplots()
    fc1_dis_shape = model_fc1.shape  # 2, 4, 31

    text_e = plt.text(max_x, max_x, 'Epoch 0')

    set_dots = []
    for i in range(fc1_dis_shape[0]):  # aug, kaldi
        dots = []
        for j in range(fc1_dis_shape[0]):  # weight, bias
            dot_x = model_fc1[i][j * 2][0]
            dot_y = model_fc1[i][j * 2 + 1][0]
            dot, = plt.plot(dot_x, dot_y, color=cValue_1[j], marker=marker[i])
            # text_p = plt.text(dot_x, dot_y, '%d' % j)
            dots.append(dot)
        set_dots.append(dots)

    plt.legend(['kaldi_weight', 'weight_bias', 'aug_weight', 'aug_bias'], loc='lower right', scatterpoints=1)

    def gen_dot():
        for i in range(fc1_dis_shape[2]):
            text_e.set_text('Epoch %2s' % str(i))
            yield model_fc1[:, :, i]  # 2,4

    def update_dot(newd):
        # pdb.set_trace()
        for i in range(fc1_dis_shape[0]):
            dots = set_dots[i]
            for j in range(fc1_dis_shape[0]):
                dot = dots[j]

                dot_x = newd[i][j * 2]
                dot_y = newd[i][j * 2 + 1]

                dot.set_data(dot_x, dot_y)
                # text_p.set_position((dot_x, dot_y))
        return set_dots

    ani = animation.FuncAnimation(fig, update_dot, frames=gen_dot, interval=800)
    ani.save(args.extract_path + "/fc1.gif", writer='pillow', fps=2)

    # for i in range(fc1_dis_shape[0]):
    #     x = model_fc1[i][i * 2]
    #     y = model_fc1[i][i * 2 + 1]
    #     plt.plot(x, y, marker=marker[i])
    #
    # plt.legend(['kaldi', 'aug'], loc='upper right')
    # plt.savefig(args.extract_path + "/fc1.png")
    print('Saving %s' % args.extract_path + "/fc1.gif")


if __name__ == '__main__':
    main()
