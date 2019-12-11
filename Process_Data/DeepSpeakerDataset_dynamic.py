#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function

import os
import pathlib
import random

import numpy as np
import pdb

import torch.utils.data as data
from torch.utils.data import Dataset


def find_classes(voxceleb):
    classes = list(set([datum['speaker_id'] for datum in voxceleb]))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def find_speakers(voxceleb):
    classes = list(set([datum['subset'] for datum in voxceleb]))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def create_indices(_features):
    inds = dict()
    for idx, (feature_path,label) in enumerate(_features):
        if label not in inds:
            inds[label] = []
        inds[label].append(feature_path)
    return inds

def generate_triplets_call(indices, n_classes):
    """
    :param indices: {spks: wavs,...]
    :param n_classes: len(spks)
    :return: troplets group: class1_wav1, class1_wav2, class2_wav1, class1, class2
    """

    # Indices = array of labels and each label is an array of indices
    #indices = create_indices(features)

    c1 = np.random.randint(0, n_classes)
    c2 = np.random.randint(0, n_classes)
    while len(indices[c1]) < 2:
        c1 = np.random.randint(0, n_classes)

    while c1 == c2:
        c2 = np.random.randint(0, n_classes)
    if len(indices[c1]) == 2:  # hack to speed up process
        n1, n2 = 0, 1
    else:
        n1 = np.random.randint(0, len(indices[c1]) - 1)
        n2 = np.random.randint(0, len(indices[c1]) - 1)
        while n1 == n2:
            n2 = np.random.randint(0, len(indices[c1]) - 1)
    if len(indices[c2]) ==1:
        n3 = 0
    else:
        n3 = np.random.randint(0, len(indices[c2]) - 1)

    return ([indices[c1][n1], indices[c1][n2], indices[c2][n3],c1,c2])

class DeepSpeakerDataset(data.Dataset):
    """
    This dataset class is for triplet training.
    """

    def __init__(self, voxceleb, dir, n_triplets,loader, transform=None, *arg, **kw):

        print('Looking for audio [wav] files in {}.'.format(dir))

        if len(voxceleb) == 0:
            raise(RuntimeError(('This is not data in the dataset')))

        for vox_item in voxceleb:

            for ky in vox_item:
                if isinstance(vox_item[ky], np.bytes_):
                    vox_item[ky] = vox_item[ky].decode('utf-8')

        classes, class_to_idx = find_classes(voxceleb)
        features = []

        for vox_item in voxceleb:
            item = (dir + "/" + vox_item['filename']+'.wav', class_to_idx[vox_item['speaker_id']])
            features.append(item)

        self.root = dir
        #self.features = features
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader

        self.n_triplets = n_triplets

        #print('Generating {} triplets'.format(self.n_triplets))
        self.indices = create_indices(features)



    def __getitem__(self, index):
        '''

        Args:
            index: Index of the triplet or the matches - not of a single feature

        Returns:

        '''
        def transform(feature_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
            """
            feature = self.loader(feature_path)
            return self.transform(feature)

        # Get the index of each feature in the triplet
        a, p, n, c1, c2 = generate_triplets_call(self.indices, len(self.classes))
        # transform features if required
        feature_a, feature_p, feature_n = transform(a), transform(p), transform(n)
        return feature_a, feature_p, feature_n, c1, c2

    def __len__(self):
        return self.n_triplets

class DeepSpeakerEnrollDataset(data.Dataset):

    def __init__(self, audio_set, dir, loader, enroll=True, transform=None, *arg, **kw):

        print('Looking for audio [wav/npy] files in {}.'.format(dir))

        if len(audio_set) == 0:
            raise(RuntimeError(('This is not data in the dataset for path: {}'.format(dir))))

        #classes, class_to_idx = find_classes(audio_set)
        self.root = dir
        self.enroll = enroll
        classes, class_to_idx = find_speakers(audio_set)
        features = []
        uttids = []
        for index, vox_item in enumerate(audio_set):
            feat_item = (vox_item['filename']+'.wav', class_to_idx[vox_item['subset']])
            uttid_item = (vox_item['utt_id'], index)

            features.append(feat_item)
            uttids.append(uttid_item)
        self.uttid = uttids
        self.features = features
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader

        self.indices = create_indices(features)


    def __getitem__(self, index):
        '''
        Args:
            index: Index of the triplet or the matches - not of a single feature
        Returns:
        '''
        def transform(feature_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
            """
            feature = self.loader(feature_path)
            return self.transform(feature)

        # Get the index of feature in the indices
        feature = self.features[index]
        feature = transform(feature[0])
        if self.enroll:
            label = self.features[index][1]
        else:
            label = self.uttid[index][1]

        return feature, label

    def __len__(self):
        return len(self.features)

class ClassificationDataset(data.Dataset):
    def __init__(self, voxceleb, dir, loader, transform=None, return_uid=False, *arg, **kw):
        print('Looking for audio [npy] features files in {}.'.format(dir))
        if len(voxceleb) == 0:
            raise(RuntimeError(('This is not data in the dataset')))
        for vox_item in voxceleb:

            for ky in vox_item:
                if isinstance(vox_item[ky], np.bytes_):
                    vox_item[ky] = vox_item[ky].decode('utf-8')

        classes, class_to_idx = find_classes(voxceleb)
        features = []
        # pdb.set_trace()
        null_spks = []
        for vox_item in voxceleb:

            # for ky in vox_item:
            #     if isinstance(vox_item[ky], np.bytes_):
            #         vox_item[ky] = vox_item[ky].decode('utf-8')

            vox_path = dir + "/" + vox_item['filename']+'.npy'
            if not os.path.exists(vox_path):
                pdb.set_trace()
                vox_path_item = pathlib.Path(vox_path)
                null_spks.append(vox_path_item.parent.parent.name)

            item = (dir + "/" + str(vox_item['filename'])+'.wav', class_to_idx[vox_item['speaker_id']])
            features.append(item)

        null_spks = list(set(null_spks))
        null_spks.sort()
        if len(null_spks) != 0:
            print('{} of speaker feats are missing!'.format(len(null_spks)))
            print(null_spks)
            exit(1)

        self.root = dir
        self.features = features
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader
        #print('Generating {} triplets'.format(self.n_triplets))
        self.indices = create_indices(features)
        self.return_uid = return_uid

    def __getitem__(self, index):
        '''

        Args:
            index: Index of the triplet or the matches - not of a single feature
        Returns:

        '''
        def transform(feature_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
            """
            feature = self.loader(feature_path)
            return self.transform(feature)

        # Get the index of feature
        feature = self.features[index][0]
        label = self.features[index][1]

        # transform features if required
        feature = transform(feature)
        if self.return_uid:
            # pdb.set_trace()
            return feature, label, self.features[index][0]

        return feature, label

    def __len__(self):
        return len(self.features)

class ValidationDataset(data.Dataset):
    '''
    Validation set should be inited by class to index list.
    '''
    def __init__(self, voxceleb, dir, loader, class_to_idx, transform=None, *arg, **kw):
        print('Looking for audio [npy] features files in {}.'.format(dir))
        if len(voxceleb) == 0:
            raise(RuntimeError(('This is not data in the dataset')))
        if len(class_to_idx) == 0:
            raise (RuntimeError(('This is no speakers in the dataset')))
        for vox_item in voxceleb:
            for ky in vox_item:
                if isinstance(vox_item[ky], np.bytes_):
                    vox_item[ky] = vox_item[ky].decode('utf-8')

        self.class_to_idx = class_to_idx
        features = []
        spks = []
        null_spks = []

        for vox_item in voxceleb:

            vox_path = dir + "/" + vox_item['filename']+'.npy'
            if not os.path.exists(vox_path):
                pdb.set_trace()
                vox_path_item = pathlib.Path(vox_path)
                null_spks.append(vox_path_item.parent.parent.name)

            item = (dir + "/" + str(vox_item['filename'])+'.wav', self.class_to_idx[vox_item['speaker_id']])
            spks.append(vox_item['speaker_id'])

            features.append(item)

        null_spks = list(set(null_spks))
        spks = list(set(spks))
        print('There are {} speakers in validation set.'.format(len(spks)))

        null_spks.sort()
        if len(null_spks) != 0:
            print('{} of speaker feats are missing!'.format(len(null_spks)))
            print(null_spks)
            exit(1)

        self.root = dir
        self.features = features
        self.classes = spks
        self.transform = transform
        self.loader = loader
        #print('Generating {} triplets'.format(self.n_triplets))

    def __getitem__(self, index):
        '''
        Args:
            index: Index of the triplet or the matches - not of a single feature
        Returns:
        '''
        def transform(feature_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
            """
            feature = self.loader(feature_path)
            return self.transform(feature)

        # Get the index of feature
        feature = self.features[index][0]
        label = self.features[index][1]

        # transform features if required
        feature = transform(feature)
        return feature, label

    def __len__(self):
        return len(self.features)


class SpeakerTrainDataset(Dataset): #定义pytorch的训练数据及类
    def __init__(self, dataset, dir, loader, transform, samples_per_speaker=8):#每个epoch每个人的语音采样数
        self.dataset = dataset
        self.dir = dir
        current_sid = -1

        # pdb.set_trace()
        self.classes = [i for i in dataset]
        self.classes.sort()
        self.class_to_idx = {self.classes[i]:i for i in range(len(self.classes))}
        self.index_to_classes = {i:self.classes[i] for i in range(len(self.classes))}

        self.n_classes = len(self.dataset)
        self.samples_per_speaker = samples_per_speaker

        self.transform = transform
        self.loader = loader

    def __len__(self):
        return self.samples_per_speaker * self.n_classes#返回一个epoch的采样数

    def __getitem__(self, sid):#定义采样方式，sid为说话人id
        sid %= self.n_classes
        spk = self.index_to_classes[sid]
        utts = self.dataset[spk]

        uid = random.randrange(0, len(utts))

        def transform(feature_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
            """
            feature = self.loader(self.dir + '/' + feature_path + '.npy')
            return self.transform(feature)

        # Get the index of feature
        feature = utts[uid]
        label = sid

        # transform features if required
        feature = transform(feature)

        return feature, label
