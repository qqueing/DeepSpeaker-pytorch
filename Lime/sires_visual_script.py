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
# !/usr/bin/env python
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
parser.add_argument('--extract-path', default='Data/extract/SiResNet34/soft/aug',
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


def main():
    epochs = np.arange(0, 31)
    conv1s = np.array([]).reshape((0, 16, 3, 3))
    grads = np.array([]).reshape((0, 2, 64))

    for i in epochs:
        save_path = pathlib.Path(args.extract_path + '/epoch_%d' % i)
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

    # pdb.set_trace()
    fig = plt.figure(figsize=(8, 8))
    plt.title('Convergence of 16 Filters')

    max_x = np.max(means)
    min_x = np.min(means)
    max_y = np.max(stds)
    min_y = np.min(stds)

    plt.xlim(min_x - 0.1 * np.abs(max_x), max_x + 0.1 * np.abs(max_x))
    plt.ylim(min_y - 0.1 * np.abs(max_y), max_y + 0.1 * np.abs(max_y))
    plt.xlabel('Means of Abs')
    plt.ylabel('Std')

    # fig, ax = plt.subplots()
    cValue_1 = ['purple', 'green', 'blue', 'pink', 'brown', 'red', 'teal', 'orange', 'magenta', 'yellow', 'grey',
                'violet', 'turquoise', 'lavender', 'tan', 'cyan', 'aqua', 'maroon', 'olive', 'salmon', 'beige', 'lilac',
                'black', 'peach', 'lime', 'indigo', 'mustard', 'rose', 'aquamarine', 'navy', 'gold', 'plum', 'burgundy',
                'khaki', 'taupe', 'chartreuse', 'mint', 'sand', 'puce', 'seafoam', 'goldenrod', 'slate', 'rust',
                'cerulean', 'ochre', 'crimson', 'fuchsia', 'puke', 'eggplant', 'white', 'sage', 'brick', 'cream',
                'coral', 'greenish', 'grape', 'azure', 'wine', 'cobalt', 'pinkish', 'vomit', 'moss', 'grass',
                'chocolate', 'cornflower', 'charcoal', 'pumpkin', 'tangerine', 'raspberry', 'orchid', 'sky']
    dots = []
    text_e = plt.text(max_x, max_y, 'Epoch 0')
    for i in range(len(means[0])):
        dot, = plt.plot(means[0][i], stds[0][i], color=cValue_1[i], marker='o')
        text_p = plt.text(means[0][i], stds[0][i], '%d' % i)
        dots.append([dot, text_p])

    def gen_dot():
        for i in range(len(means)):
            text_e.set_text('Epoch %2s' % str(i))
            newdot = [means[i], stds[i]]
            yield newdot

    def update_dot(newd):
        # pdb.set_trace()
        for i in range(len(means[0])):
            dot, text_p = dots[i]
            dot.set_data(newd[0][i], newd[1][i])
            text_p.set_position((newd[0][i], newd[1][i]))

        return dots

    ani = animation.FuncAnimation(fig, update_dot, frames=gen_dot, interval=800)
    ani.save(args.extract_path + "/conv1s.gif", writer='pillow', fps=2)

    fig = plt.figure(figsize=(8, 8))
    plt.title('Filting over 8000Hz, 0-20 Epochs')
    plt.xlabel('Frequency')
    plt.ylabel('Weight')

    mel_high = 2595 * np.log10(1 + 8000 / 700)
    mel_cen = [mel_high / 65 * i for i in range(1, 65)]
    mel_cen = np.array(mel_cen)
    x = 700 * (10 ** (mel_cen / 2595) - 1)

    y = np.nan_to_num(grads)
    # pdb.set_trace()
    max_x = np.max(x)
    min_x = np.min(x)
    max_y = np.max(y)
    min_y = np.min(y)
    plt.xlim(min_x - 0.1 * np.abs(max_x), max_x + 0.1 * np.abs(max_x))
    plt.ylim(min_y - 0.1 * np.abs(max_y), max_y + 0.1 * np.abs(max_y))
    # pdb.set_trace()
    # print(y.shape)
    text_e = plt.text(min_x, max_y, 'Epoch 0')

    dots = []
    for i in range(len(y[0])):
        dot, = plt.plot(x, y[0][i], color=cValue_1[i])
        dots.append(dot)
    plt.legend(['Train_set', 'Valid set'], loc='upper right')

    def gen_line():
        for i in range(1, len(y)):
            newdot = [x, y[i]]
            text_e.set_text('Epoch %2s' % str(i))
            yield newdot

    def update_line(newd):
        for i in range(2):
            dots[i].set_data(newd[0], newd[1][i])
        return dots

    ani = animation.FuncAnimation(fig, update_line, frames=gen_line, interval=800)
    ani.save(args.extract_path + "/grads.gif", writer='pillow', fps=2)
    # plt.show()


if __name__ == '__main__':
    main()
