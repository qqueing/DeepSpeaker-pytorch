#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Cosine.py
@Time: 19-6-26 下午9:43
@Overview: Implement Cosine Score for speaker identification!
Enrollment set files will be in the 'Data/enroll_set.npy' and the classes-to-index file is 'Data/enroll_classes.npy'
Test set files are in the 'Data/test_set.npy' and the utterances-to-index file is 'Data/test_classes.npy'
"""
import numpy as np
import torch.nn as nn

ENROLL_FILE = "Data/xvector/enroll/extract_adagrad-lr0.1-wd0.0-embed512-alpha10.npy"
ENROLL_CLASS = "Data/enroll_classes.npy"
TEST_FILE = "Data/xvector/test/extract_adagrad-lr0.1-wd0.0-embed512-alpha10.npy"
TEST_CLASS = "Data/test_classes.npy"

# test_vec = np.array([1,2,3,4])
# refe_vec = np.array([8,3,3,4])

def normalize(narray, order=2, axis=1):
    norm = np.linalg.norm(narray, ord=order, axis=axis, keepdims=True)
    return(narray/norm + np.finfo(np.float32).eps)

def cos_dis(test_vec, refe_vec):
    vec1 = normalize(test_vec, axis=0)
    vec2 = normalize(refe_vec, axis=0)
    dis = np.matmul(vec1, vec2.T)
    return(dis)

enroll_features = np.load(ENROLL_FILE, allow_pickle=True)
enroll_classes = np.load(ENROLL_CLASS, allow_pickle=True).item()
test_features = np.load(TEST_FILE, allow_pickle=True)
test_classes = np.load(TEST_CLASS, allow_pickle=True)
enroll_dict = dict()
for item in enroll_classes:
    num=0
    feat = np.zeros([512], dtype=float)
    for (label, feature) in enroll_features:
        if label==enroll_classes[item]:
            feat += feature.detach().numpy()
            num += 1
    enroll_dict[item] = feat/num

similarity = {}
for (label, feature) in test_features:
    utt = {}
    for item in enroll_dict:
        utt[item] = np.linalg.norm(feature.detach().numpy()-enroll_dict[item])

    for utterance in test_classes:
        if int(utterance[1])==label.item():
            test_id = utterance[0]
    similarity[test_id]=utt
print(similarity)

# cos_dis(test_vec, refe_vec)

