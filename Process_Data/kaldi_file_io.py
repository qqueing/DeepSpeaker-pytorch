#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: kaldi_file_io.py
@Time: 2019/12/10 下午9:28
@Overview:
"""
import os
import pathlib
import pdb
import random

from tqdm import tqdm

import Process_Data.constants as c
import kaldi_io
import numpy as np
import torch.utils.data as data

class KaldiTrainDataset(data.Dataset):
    def __init__(self, dir, samples_per_speaker, transform):

        feat_scp = dir + '/feats.scp'
        spk2utt = dir + '/spk2utt'
        utt2spk = dir + '/utt2spk'
        num_valid = 5

        if not os.path.exists(feat_scp):
            raise FileExistsError(feat_scp)
        if not os.path.exists(spk2utt):
            raise FileExistsError(spk2utt)

        dataset = {}
        with open(spk2utt, 'r') as u:
            all_cls = u.readlines()
            for line in all_cls:
                spk_utt = line.split(' ')
                spk_name = spk_utt[0]
                if spk_name not in dataset.keys():
                    spk_utt[-1]=spk_utt[-1].rstrip('\n')
                    dataset[spk_name] = spk_utt[1:]
        utt2spk_dict = {}
        with open(utt2spk, 'r') as u:
            all_cls = u.readlines()
            for line in all_cls:
                utt_spk = line.split(' ')
                uid = utt_spk[0]
                if uid not in utt2spk_dict.keys():
                    utt_spk[-1] = utt_spk[-1].rstrip('\n')
                    utt2spk_dict[uid] = utt_spk[-1]
        # pdb.set_trace()

        speakers = [spk for spk in dataset.keys()]
        speakers.sort()
        print('==>There are {} speakers in Dataset.'.format(len(speakers)))
        spk_to_idx = {speakers[i]: i for i in range(len(speakers))}
        idx_to_spk = {i: speakers[i] for i in range(len(speakers))}

        uid2feat = {}  # 'Eric_McCormack-Y-qKARMSO7k-0001.wav': feature[frame_length, feat_dim]
        pbar = tqdm(enumerate(kaldi_io.read_mat_scp(feat_scp)))
        for idx, (utt_id, feat) in pbar:
            uid2feat[utt_id] = feat

        print('==>There are {} utterances in Train Dataset.'.format(len(uid2feat)))
        valid_set = {}
        valid_uid2feat = {}
        valid_utt2spk_dict = {}

        for spk in speakers:
            if spk not in valid_set.keys():
                valid_set[spk] = []
                for i in range(num_valid):
                    if len(dataset[spk]) <= 1:
                        break
                    j = np.random.randint(len(dataset[spk]))
                    utt = dataset[spk].pop(j)
                    valid_set[spk].append(utt)

                    valid_uid2feat[valid_set[spk][-1]] = uid2feat.pop(valid_set[spk][-1])
                    valid_utt2spk_dict[utt] = utt2spk_dict[utt]

        print('==>Spliting {} utterances for Validation.\n'.format(len(valid_uid2feat)))

        self.feat_dim = uid2feat[dataset[speakers[0]][0]].shape[1]
        self.speakers = speakers
        self.dataset = dataset
        self.valid_set = valid_set
        self.valid_uid2feat = valid_uid2feat
        self.valid_utt2spk_dict = valid_utt2spk_dict
        self.uid2feat = uid2feat
        self.spk_to_idx = spk_to_idx
        self.idx_to_spk = idx_to_spk
        self.num_spks = len(speakers)
        self.transform = transform
        self.samples_per_speaker = samples_per_speaker

    def __getitem__(self, sid):
        sid %= self.num_spks
        spk = self.idx_to_spk[sid]
        utts = self.dataset[spk]
        n_samples = 0
        y = np.array([[]]).reshape(0, self.feat_dim)

        frames = c.N_SAMPLES
        while n_samples < frames:

            uid = random.randrange(0, len(utts))
            feature = self.uid2feat[utts[uid]]

            # Get the index of feature
            if n_samples == 0:
                start = int(random.uniform(0, len(feature)))
            else:
                start = 0
            stop = int(min(len(feature) - 1, max(1.0, start + frames - n_samples)))
            try:
                y = np.concatenate((y, feature[start:stop]), axis=0)
            except:
                pdb.set_trace()
            n_samples = len(y)
            # transform features if required

        feature = self.transform(y)
        label = sid
        return feature, label

    def __len__(self):
        return self.samples_per_speaker * len(self.speakers)  # 返回一个epoch的采样数


class KaldiValidDataset(data.Dataset):
    def __init__(self, valid_set, spk_to_idx, valid_uid2feat, valid_utt2spk_dict, transform):

        speakers = [spk for spk in valid_set.keys()]
        speakers.sort()
        self.speakers = speakers
        self.dataset = valid_set
        self.valid_set = valid_set
        self.uid2feat = valid_uid2feat
        self.utt2spk_dict = valid_utt2spk_dict
        self.spk_to_idx = spk_to_idx
        self.num_spks = len(speakers)
        self.transform = transform

    def __getitem__(self, index):
        uid = list(self.uid2feat.keys())[index]
        spk = self.utt2spk_dict[uid]

        feature = self.transform(self.uid2feat[uid])
        label = self.spk_to_idx[spk]

        return feature, label

    def __len__(self):
        return len(self.uid2feat)


class KaldiTestDataset(data.Dataset):
    def __init__(self, dir, transform):

        feat_scp = dir + '/feats.scp'
        spk2utt = dir + '/spk2utt'
        trials = dir + '/trials'

        if not os.path.exists(feat_scp):
            raise FileExistsError(feat_scp)
        if not os.path.exists(spk2utt):
            raise FileExistsError(spk2utt)
        if not os.path.exists(trials):
            raise FileExistsError(trials)

        dataset = {}
        with open(spk2utt, 'r') as u:
            all_cls = u.readlines()
            for line in all_cls:
                spk_utt = line.split(' ')
                spk_name = spk_utt[0]
                if spk_name not in dataset.keys():
                    spk_utt[-1] = spk_utt[-1].rstrip('\n')
                    dataset[spk_name] = spk_utt[1:]

        speakers = [spk for spk in dataset.keys()]
        speakers.sort()
        print('==>There are {} speakers in Test Dataset.'.format(len(speakers)))

        uid2feat = {}
        for utt_id, feat in kaldi_io.read_mat_scp(feat_scp):
            uid2feat[utt_id] = feat
        print('==>There are {} utterances in Test Dataset.'.format(len(uid2feat)))

        trials_pair = []
        with open(trials, 'r') as t:
            all_pairs = t.readlines()
            for line in all_pairs:
                pair = line.split(' ')
                if pair[2] == 'nontarget\n':
                    pair_true = False
                else:
                    pair_true = True

                trials_pair.append((pair[0], pair[1], pair_true))

        print('==>There are {} pairs in test Dataset.\n'.format(len(trials_pair)))

        self.feat_dim = uid2feat[dataset[speakers[0]][0]].shape[1]
        self.speakers = speakers
        self.uid2feat = uid2feat
        self.trials_pair = trials_pair
        self.num_spks = len(speakers)
        self.transform = transform

    def __getitem__(self, index):
        uid_a, uid_b, label = self.trials_pair[index]

        data_a = self.uid2feat[uid_a]
        data_b = self.uid2feat[uid_b]

        data_a = self.transform(data_a)
        data_b = self.transform(data_b)

        return data_a, data_b, label

    def __len__(self):
        return len(self.trials_pair)


def write_xvector_ark(uid, xvector, write_path, set):
    """

    :param uid: generated by dataset class
    :param xvector:
    :param write_path:
    :param set: train or test
    :return:
    """
    file_path = pathlib.Path(write_path)
    if not file_path.exists():
        os.makedirs(str(file_path))

    ark_file = write_path+'/{}_xvector.ark'.format(set)
    scp_file = write_path+'/{}_xvector.scp'.format(set)

    # write scp and ark file
    with open(scp_file, 'w') as scp, open(ark_file, 'wb') as ark:
        for i in range(len(uid)):
            vec = xvector[i]
            len_vec = len(vec.tobytes())
            key = uid[i]

            kaldi_io.write_vec_flt(ark, vec, key=key)
            # print(ark.tell())
            scp.write(str(uid[i]) + ' ' + str(ark_file) + ':' + str(ark.tell()-len_vec-10) + '\n')

    print('\nark,scp files are in: {}, {}.'.format(ark_file, scp_file))

    # Prepare utt2spk file
    if set=='train':
        utt2spk_file = write_path+'/utt2spk'
        with open(utt2spk_file, 'w') as utt2spk:
            for i in range(len(uid)):
                spk = uid[i].split('-')[0]
                utt2spk.write(str(uid[i]) + ' ' + str(spk)+'\n')

        print('utt2spk file is in: {}.'.format(utt2spk_file))

# uid = ['A.J._Buckley-1zcIwhmdeo4-0001.wav', 'A.J._Buckley-1zcIwhmdeo4-0002.wav', 'A.J._Buckley-1zcIwhmdeo4-0003.wav', 'A.J._Buckley-7gWzIy6yIIk-0001.wav']
# xvector = np.random.randn(4, 512).astype(np.float32)
#
# ark_file = '../Data/xvector.ark'
# scp_file = '../Data/xvector.scp'
#
# write_xvector_ark(uid, xvector, ark_file, scp_file)

