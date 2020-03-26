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

parser.add_argument('--check-path', default='Data/checkpoint/SuResCNN10/spect/aug',
                    help='folder to output model checkpoints')
parser.add_argument('--extract-path', default='Data/extract/SuResCNN10/spect',
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
            'violet', 'turquoise', 'lavender', 'tan', 'cyan', 'aqua', 'maroon', 'olive', 'salmon', 'beige',
            'black', 'peach', 'lime', 'indigo', 'mustard', 'rose', 'aquamarine', 'navy', 'gold', 'plum', 'burgundy',
            'khaki', 'taupe', 'chartreuse', 'mint', 'sand', 'puce', 'seafoam', 'goldenrod', 'slate', 'rust',
            'cerulean', 'ochre', 'crimson', 'fuchsia', 'puke', 'eggplant', 'white', 'sage', 'brick', 'cream',
            'coral', 'greenish', 'grape', 'azure', 'wine', 'cobalt', 'pinkish', 'vomit', 'moss', 'grass',
            'chocolate', 'cornflower', 'charcoal', 'pumpkin', 'tangerine', 'raspberry', 'orchid', 'sky']
marker = ['o', 'x']


def main():
    # conv1s = np.array([]).reshape((0, 64, 5, 5))
    # grads = np.array([]).reshape((0, 2, 161))
    model_set = ['kaldi_5wd', 'aug']
    epochs = np.arange(0, 31)

    if os.path.exists(args.extract_path + '/mean_grads.npy'):
        input_grads = np.load(args.extract_path + '/mean_grads.npy')
    else:
        input_grads = []

        for m in model_set:
            extract_paths = os.path.join(args.extract_path, m)
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
                grads_means = np.array([]).reshape((0, 161))

                for name in ['train', 'valid']:
                    sets_files = list(save_path.glob('vox1_%s.*.bin' % name))
                    grad_mean = np.zeros((161))
                    num_utt = 0
                    for f in sets_files:
                        with open(str(f), 'rb') as f:
                            sets = pickle.load(f)
                            for (uid, orig, conv1, bn1, relu1, grad) in sets:
                                grad_mean += np.mean(grad, axis=0)
                                num_utt += 1
                    grads_means = np.concatenate((grads_means, grad_mean[np.newaxis, :] / num_utt), axis=0)

                grads_means = grads_means[np.newaxis, :]
                grads = np.concatenate((grads, grads_means), axis=0)

            input_grads.append(grads)

        while input_grads[0].shape[0] < input_grads[1].shape[0]:
            input_grads[0] = np.concatenate(
                (input_grads[0], input_grads[0][-1, :, :].reshape(1, input_grads[0].shape[1], input_grads[0].shape[2])),
                axis=0)

        input_grads = np.array(input_grads)  # 2,21,64
        np.save(args.extract_path + '/mean_grads.npy', input_grads)

    # pdb.set_trace()
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 8))
    plt.title('Filting over 8000Hz')
    plt.xlabel('Frequency')
    plt.ylabel('Weight')

    x = np.arange(161) * 8000 / 161  # [0-8000]
    y = np.nan_to_num(input_grads)  # 2,
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
    ani.save(args.extract_path + "/mean_grads.gif", writer='pillow', fps=2)


if __name__ == '__main__':
    main()
