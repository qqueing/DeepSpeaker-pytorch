#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: XvectorDataset.py
@Time: 2019/9/19 上午11:51
@Overview:
"""
import torch as t
import os
from torch.utils import data
import sidekit
import numpy as np
import torchvision.transforms as transforms

class TrainSet(data.Dataset):

    def __init__(self, root):
        speakers_dir = os.listdir(root)
        self.speakers = len(speakers_dir)
        self.speakers_dir = np.asarray(speakers_dir)
        for i in range(len(speakers_dir)):
            speakers_dir[i] = root + speakers_dir[i]
        speech = []
        for i in speakers_dir:
            speech_dir = os.listdir(i)
            for j in range(20, len(speech_dir)):
                speech.append(speech_dir[j].split('.')[0])
        for i in range(len(speech)):
            speech[i] = speech[i].split('_')[0] + '/' + speech[i]

        self.speech = np.asarray(speech)

    def __getitem__(self, index):

        features_server = sidekit.FeaturesServer(features_extractor = None,
                                                 feature_filename_structure = '../all_feature/{}.h5',
                                                 sources = None,
                                                 dataset_list = ['fb'],
                                                 mask = None,
                                                 feat_norm = 'cms',
                                                 global_cmvn = None,
                                                 dct_pca = False,
                                                 dct_pca_config = None,
                                                 sdc = False,
                                                 sdc_config = None,
                                                 delta = False,
                                                 double_delta = False,
                                                 delta_filter = None,
                                                 context = None,
                                                 traps_dct_nb = None,
                                                 rasta = True,
                                                 keep_all_features = False)

        show_list = self.speech[index]
        speaker = show_list.split('/')[0]
        features, _ = features_server.load(show_list, channel = 0)
        features = features.astype(np.float32)
        ind = np.argwhere(self.speakers_dir == speaker)[0]
        label = ind.astype(np.int64)[0] #这里只要指出label所在的索引就好了，比如是第20个说话人说的，那么label就是[20]
        features = features.reshape(1, features.shape[1], features.shape[0])
        features = t.tensor(features)
        img = transforms.ToPILImage()(features)
        features = transforms.Resize((24, 400))(img)
        features = transforms.ToTensor()(features)

        return features.view(features.size()[1], features.size()[2]), label

    def __len__(self):
        return len(self.speech)