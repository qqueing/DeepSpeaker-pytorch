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
import pickle
import numpy as np
import pathlib

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

    for i in epochs:
        save_path = pathlib.Path(args.extract_path + '/epoch_%d' % i)
        trains = list(save_path.glob('vox1_train.*.bin'))
        with open(str(trains[0])) as f:
            utts_info = pickle.load(f)
            for (uid, orig, conv1, bn1, relu1, grad) in utts_info:
                print(uid)


if __name__ == '__main__':
    main()
