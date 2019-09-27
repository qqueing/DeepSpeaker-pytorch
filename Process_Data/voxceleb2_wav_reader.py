#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: voxceleb2_wav_reader.py
@Time: 2019/9/19 下午7:54
@Overview:
"""
import os
from glob import glob
import pathlib
import numpy as np
import sys
import re

def parse_txt(txt):
    lines = [line.strip() for line in open(txt, 'r').readlines()]
    speaker = lines[0].split('\t')[-1]
    uri = lines[1].split('\t')[-1]
    duration = float(lines[2].split('\t')[-1].split()[0])
    subset = lines[3].split('\t')[-1]

    file_list = []
    for line in lines[5:]:
        file_location, start, end = line.split()
        file_list.append(file_location)


    return subset, uri, speaker, file_list



def find_files(directory, pattern='*/*/*/*.wav'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def read_voxceleb2_structure(directory):
    voxceleb2 = []

    data_root = pathlib.Path(directory)
    data_root.cwd()
    print('>>Data root is %s' % str(data_root))

    # all the paths of wav files
    # dev/acc/spk_id/utt_group/wav_id.wav
    all_dev_path = list(data_root.glob('*/*/*/*/*.wav'))
    all_test_path = list(data_root.glob('*/*/*/*.m4a'))

    # print(str(pathlib.Path.relative_to(all_wav_path[0], all_wav_path[0].parents[4])).rstrip('.wav'))

    all_dev_path = [str(pathlib.Path.relative_to(path, path.parents[4])).rstrip(".wav") for path in all_dev_path]
    all_test_path = [str(pathlib.Path.relative_to(path, path.parents[3])).rstrip(".m4a") for path in all_test_path]
    all_wav_path = np.concatenate((all_dev_path, all_test_path))

    dev_subset = ['dev' for path in all_dev_path]
    test_subset = ['test' for path in all_test_path]
    subset = np.concatenate((dev_subset, test_subset))

    dev_speaker = [pathlib.Path(path).parent.parent.name for path in all_dev_path]
    test_speaker = [pathlib.Path(path).parent.parent.name for path in all_test_path]
    speaker = np.concatenate((dev_speaker, test_speaker))

    all_wav = np.transpose([all_wav_path, subset, speaker])

    for file in all_wav:
        voxceleb2.append({'filename': file[0], 'speaker_id': file[2], 'uri': 0, 'subset': file[1]})
        # print(str(file[0]))
        # exit()

    num_speakers = len(set([datum['speaker_id'] for datum in voxceleb2]))
    print('>>Found {} files with {} different speakers.'.format(str(len(voxceleb2)), str(num_speakers)))
    #print(voxceleb.head(10))
    return voxceleb2

def read_extract_audio(directory):
    audio_set = []

    print('>>Data root is %s' % str(directory))

    all_wav_path = []
    for dirs, dirnames, files in os.walk(directory):
        for file in files:
            # print(str(file))
            if re.match(r'^[^\.].*\.npy', str(file)) is not None:
                all_wav_path.append(str(os.path.join(dirs, file)))

    for file in all_wav_path:
        spkid = pathlib.Path(file).parent.parent.name
        audio_set.append({'filename': file.rstrip('.npy'), 'utt_id': file.rstrip('.npy'), 'uri': 0, 'subset': spkid if spkid!='Data' else 'test'})

    print('>>Found {} wav/npy files for extracting xvectors.'.format(len(audio_set)))
    #print(voxceleb.head(10))
    return audio_set

def voxceleb2_list_reader(data_path):
    """
        Check if we could resume dataset variables from local list(.npy).
    :param data_path: the dataset root
    :return: the data list
    """
    voxceleb_list = "Data/voxceleb2.npy"
    voxceleb_dev_list = "Data/voxceleb2_dev.npy"

    if os.path.isfile(voxceleb_list):
        voxceleb = np.load(voxceleb_list, allow_pickle=True)
        if len(voxceleb)!=1128246:
            print("The number of wav files may be wrong!")
    else:
        voxceleb = read_voxceleb2_structure(data_path)
        np.save(voxceleb_list, voxceleb)

    if os.path.isfile(voxceleb_dev_list):
        voxceleb_dev = np.load(voxceleb_dev_list, allow_pickle=True)
    else:
        voxceleb_dev = [datum for datum in voxceleb if datum['subset'] == 'dev']
        np.save(voxceleb_dev_list, voxceleb_dev)

    return voxceleb, voxceleb_dev


# read_my_voxceleb_structure('/data/voxceleb')




