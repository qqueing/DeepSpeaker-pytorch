#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: make_spectrogram.py
@Time: 2020/2/28 5:53 PM
@Overview: Make Spectorgrams with kaldi data format.
"""

from __future__ import print_function

import argparse
import os
import pathlib
import sys
import pdb
from multiprocessing import Process, Queue, Pool
import time
import numpy as np
from kaldi_io import kaldi_io

import Process_Data.constants as c
from Process_Data.audio_processing import Make_Spect

parser = argparse.ArgumentParser(description='Computing spectrogram!')

parser.add_argument('--nj', type=int, default=12, metavar='E',
                    help='number of jobs to make feats (default: 10)')
parser.add_argument('--data-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1/dev',
                    help='number of jobs to make feats (default: 10)')
parser.add_argument('--out-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev',
                    help='number of jobs to make feats (default: 10)')

parser.add_argument('--conf', type=str, default='condf/spect.conf', metavar='E',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--vad-proportion-threshold', type=float, default=0.12, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--vad-frames-context', type=int, default=2, metavar='E',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()


def compute_wav_path(data_path):
    wav_scp = os.path.join(data_path, 'wav.scp')
    feat_scp = os.path.join(data_path, 'feat.scp')
    feat_ark = os.path.join(data_path, 'feat.ark')

    assert os.path.exists(wav_scp)

    f = open(wav_scp, 'r')
    wav_scp_lsts = f.readlines()
    f.close()

    uid2path = {}
    for line in wav_scp_lsts:
        l = line.split()
        uid2path[l[0]] = l[1]

    with open(feat_scp, 'w') as scp, open(feat_ark, 'wb') as ark:
        for uid in uid2path.keys():
            feat = Make_Spect(wav_path=uid2path[uid], windowsize=0.02, stride=0.01)
            # np_fbank = Make_Fbank(filename=uid2path[uid], use_energy=True, nfilt=c.TDNN_FBANK_FILTER)

            len_vec = len(feat.tobytes())
            key = uid
            kaldi_io.write_vec_flt(ark, feat, key=key)

            scp.write(str(uid[i]) + ' ' + str(feat_scp) + ':' + str(ark.tell() - len_vec - 10) + '\n')


def compute_wav_path(wav, feat_scp, feat_ark):
    feat = Make_Spect(wav_path=wav[1], windowsize=0.02, stride=0.01)
    # np_fbank = Make_Fbank(filename=uid2path[uid], use_energy=True, nfilt=c.TDNN_FBANK_FILTER)

    len_vec = len(feat.tobytes())
    key = wav[0]
    kaldi_io.write_vec_flt(feat_ark, feat, key=key)

    feat_scp.write(str(key) + ' ' + str(feat_ark.name) + ':' + str(feat_ark.tell() - len_vec - 10) + '\n')


class MakeFeatsProcess(Process):

    def __init__(self, out_dir, item, proid, queue):
        super(MakeFeatsProcess, self).__init__()  # 重构run函数必须要写
        self.item = item
        self.proid = proid
        self.queue = queue

        #  wav_scp = os.path.join(data_path, 'wav.scp')
        feat_scp = os.path.join(out_dir, 'feat.%d.scp' % proid)
        feat_ark = os.path.join(out_dir, 'feat.%d.ark' % proid)

        self.feat_scp = open(feat_scp, 'w')
        self.feat_ark = open(feat_ark, 'wb')

    def run(self):
        for wav in self.item:
            pair = wav.split()
            compute_wav_path(pair, self.feat_scp, self.feat_ark)
            self.queue.put(pair[0])
            if self.queue.qsize() % 50 == 0:
                print('>> Process %s:' % str(self.proid) + str(self.queue.qsize()))

        print('>>Process {} finished!'.format(self.proid))


if __name__ == "__main__":

    nj = args.nj
    data_dir = args.data_dir
    out_dir = args.out_dir

    wav_scp_f = os.path.join(data_dir, 'wav.scp')
    with open(wav_scp_f, 'r') as f:
        wav_scp = f.readlines()
        assert len(wav_scp) > 0

    assert os.path.exists(data_dir)
    assert os.path.exists(wav_scp_f)

    num_utt = len(wav_scp)
    chunk = int(num_utt / nj)
    start_time = time.time()

    completed_queue = Queue()
    processpool = []

    print('Plan to make feats for %d utterances.' % num_utt)
    for i in range(0, nj):
        j = (i + 1) * chunk

        if i == (nj - 1):
            j = num_utt

        out_dir = os.path.join(out_dir, 'Split%d/%d' % (nj, i))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        p = MakeFeatsProcess(out_dir, wav_scp[i * chunk:j], i, completed_queue)
        p.start()
        processpool.append(p)

    for p in processpool:
        p.join()

    print('For multi process Completed!')

"""
For multi threads, average making seconds for 47 speakers is 4.579958657
For one threads, average making seconds for 47 speakers is 4.11888732301

For multi process, average making seconds for 47 speakers is 1.67094940328
For one process, average making seconds for 47 speakers is 3.64203325738
"""
