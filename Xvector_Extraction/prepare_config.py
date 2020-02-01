#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: prepare_config.py
@Time: 2019/12/16 3:53 PM
@Overview:
"""
import numpy

scp_file = '../Data/Fb_No/xvector.scp'
test_scp_file = '../Data/Fb_No/test_xvector.scp'

utt2spk = '../Data/Fb_No/utt2spk'
new_utt2spk = '../Data/Fb_No/new_utt2spk'
new_scp_file = '../Data/Fb_No/new_xvector.scp'
new_test_scp_file = '../Data/Fb_No/new_test_xvector.scp'


def utt2spk_from_scp2ark_path(scp_file, utt2spk):
    with open(scp_file, 'r') as f:
        with open(utt2spk, 'w') as u:
            pairs = f.readlines()
            for utt2ark in pairs:
                wav_path, ark_pos = utt2ark.split()

                u.write(wav_path + ' ' + wav_path.split('/')[-3] + '\n')

def utt2spk_from_scp2ark(scp_file, utt2spk):
    with open(scp_file, 'r') as f:
        with open(utt2spk, 'w') as u:
            pairs = f.readlines()
            for utt2ark in pairs:
                uid, ark_pos = utt2ark.split()

                u.write(uid + ' ' + uid.split('-')[0] + '\n')

def id2name(meta_file):
    spk_id_name = {}

    with open(meta_file, 'r') as f:
        spks = f.readlines()
        for spk in spks:
            items = spk.split()
            spk_id_name[items[0]] = items[1]

    return spk_id_name

def correct_uttid(ori_scp_file, spk_id_name, new_scp_file):
    with open(ori_scp_file, 'r') as o:
        with open(new_scp_file, 'w') as n:
            pairs = o.readlines()
            for utt2ark in pairs:
                wav_path, ark_pos = utt2ark.split()
                path_meta = wav_path.split('/')

                name = spk_id_name[path_meta[-3]]
                utt = path_meta[-2]
                uid = path_meta[-1][1:]

                uttid = '-'.join((name, utt, uid))
                n.write(uttid + ' ' + ark_pos + '\n')


# spk_id_name = id2name('Data/dataset/voxceleb1/vox1_meta.csv')
# correct_uttid(test_scp_file, spk_id_name, new_test_scp_file)
# utt2spk_from_scp2ark(new_scp_file, new_utt2spk)
