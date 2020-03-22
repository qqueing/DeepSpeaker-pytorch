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
import pdb
import pickle
import numpy as np
import pathlib
import matplotlib.pyplot as plt
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

parser.add_argument('--check-path', default='Data/checkpoint/SuResCNN10/spect/kaldi_5wd',
                    help='folder to output model checkpoints')
parser.add_argument('--extract-path', default='Data/extract/SuResCNN10/spect/kaldi_5wd',
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
    epochs = np.arange(0, 21)
    conv1s = np.array([]).reshape((0, 64, 5, 5))
    for i in epochs:
        save_path = pathlib.Path(args.extract_path + '/epoch_%d' % i)
        trains = list(save_path.glob('vox1_train.*.bin'))
        # with open(str(trains[0]), 'rb') as f:
        #     utts_info = pickle.load(f)
        #     for (uid, orig, conv1, bn1, relu1, grad) in utts_info:
        #         print(uid)

        cs = list(save_path.glob('model.conv1.npy'))

        conv1_epoch = np.load(str(cs[0])).squeeze()
        conv1_epoch = conv1_epoch[np.newaxis, :]
        # pdb.set_trace()
        conv1s = np.concatenate((conv1s, conv1_epoch), axis=0)

    means = np.mean(np.abs(conv1s), axis=(2, 3))
    stds = np.std(conv1s, axis=(2, 3))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel('mean')
    ax.set_ylabel('std')

    max_x = np.max(means)
    min_x = np.min(means)
    max_y = np.max(stds)
    min_y = np.min(stds)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # fig, ax = plt.subplots()
    cValue_1 = ['purple', 'green', 'blue', 'pink', 'brown', 'red', 'teal', 'orange', 'magenta', 'yellow', 'grey',
                'violet', 'turquoise', 'lavender', 'tan', 'cyan', 'aqua', 'maroon', 'olive', 'salmon', 'beige', 'lilac',
                'black', 'peach', 'lime', 'indigo', 'mustard', 'rose', 'aquamarine', 'navy', 'gold', 'plum', 'burgundy',
                'khaki', 'taupe', 'chartreuse', 'mint', 'sand', 'puce', 'seafoam', 'goldenrod', 'slate', 'rust',
                'cerulean', 'ochre', 'crimson', 'fuchsia', 'puke', 'eggplant', 'white', 'sage', 'brick', 'cream',
                'coral', 'greenish', 'grape', 'azure', 'wine', 'cobalt', 'pinkish', 'vomit', 'moss', 'grass',
                'chocolate', 'cornflower', 'charcoal', 'pumpkin', 'tangerine', 'raspberry', 'orchid', 'sky']
    dots = []
    for i in range(len(means)):
        dot, = ax.plot(means[0][i], stds[0][i], color=cValue_1[i], marker='o')
        dots.append(dot)

    def gen_dot():
        for i in range(len(means)):
            newdot = [means[i], stds[i]]
            yield newdot

    def update_dot(newd):
        # pdb.set_trace()
        for i in range(len(means)):
            dots[i].set_data(newd[0][i], newd[1][i])
            ax.annotate(str(i), (newd[0][i], newd[1][i]), fontsize=16)
        return dot

    ani = animation.FuncAnimation(fig, update_dot, frames=gen_dot, interval=800)
    ani.save("conv1s.gif", writer='pillow', fps=5)
    plt.show()


if __name__ == '__main__':
    main()
