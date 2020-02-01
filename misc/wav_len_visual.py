#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: wav_len_visual.py
@Time: 2019/12/17 1:16 PM
@Overview:
"""
import numpy as np
import matplotlib.pyplot as plt

spk_dur = '../Data/voxceleb1/spk_utt_duration.npy'

spk_dur_dic = np.load(spk_dur, allow_pickle=True).item()
all_dur = []
for key in spk_dur_dic:
    all_dur=np.concatenate((all_dur, spk_dur_dic[key]))

all_dur = all_dur.astype(np.int16)
uni_dur, counts = np.unique(all_dur, return_counts=True)

plt.bar(uni_dur*0.01, counts, width=1)
plt.figure(figsize=(41.6, 8.74))
# plt.xscale("log")
plt.show()

