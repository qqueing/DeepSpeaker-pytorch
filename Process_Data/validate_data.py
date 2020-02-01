#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: validate_data.py
@Time: 2019/9/26 上午10:51
@Overview:

Load /home/cca01/work2019/yangwenhao/mydataset/voxceleb2/fbank64/dev/aac/id02912/88v-sPZl5-w/00010.wav error!
Load /home/cca01/work2019/yangwenhao/mydataset/voxceleb2/fbank24/dev/aac/id00518/7PRhib9U-DQ/00007.wav error!
"""
from __future__ import print_function

# import os
# import pathlib
import pdb
from multiprocessing import Queue, Process
import multiprocessing

import numpy as np
from Process_Data.voxceleb2_wav_reader import voxceleb2_list_reader
from Process_Data.voxceleb_wav_reader import wav_list_reader

dataroot = '/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/Fbank64_Norm'

num_pro = 0.
skip_wav = 0.


def check_from_queue(queue, error_queue, spk_utt_duration, cpid, share_lock):
    # return np.load('/home/cca01/work2019/yangwenhao/mydataset/voxceleb2/fbank64/dev/aac/id02912/88v-sPZl5-w/00010.npy')
    while not queue.empty():
         vox2 = queue.get()

         write_path = dataroot + '/' + vox2['filename'].decode('utf-8') + '.npy'
         spk = vox2['speaker_id'].decode('utf-8')
         #print(dict(spk_utt_duration))

         # pdb.set_trace()
         try:
             item = np.load(write_path)
             #print('feat length: ' + str(len(item)))
             share_lock.acquire()
             if spk not in spk_utt_duration.keys():
                 spk_utt_duration[spk] = []
             this_spk = spk_utt_duration[spk]
             this_spk.append(len(item))
             spk_utt_duration[spk] = this_spk
             share_lock.release()
             # print('')
         except Exception:
             error_queue.put(vox2)
         #share_lock.release()
         print('\rProcess {}: There are {:6d} features left.'.format(cpid, queue.qsize()), end='\r')
    pass

def add_duration_vox(queue, error_queue, vox_duration, cpid, share_lock):
    while not queue.empty():
         vox2 = queue.get()

         write_path = dataroot + '/' + vox2['filename'].decode('utf-8') + '.npy'

         # pdb.set_trace()
         try:
             item = np.load(write_path)
             #print('feat length: ' + str(len(item)))
             share_lock.acquire()
             vox2['duration'] = len(item)
             vox_duration.append(vox2)

             share_lock.release()
             # print('')
         except Exception:
             error_queue.put(vox2)
         #share_lock.release()
         print('\rProcess {}: There are {:6d} features left.'.format(cpid, queue.qsize()), end='\r')
    pass

if __name__ == '__main__':
    queue = Queue()
    que_queue = Queue()
    # voxceleb2, voxceleb2_dev = voxceleb2_list_reader(dataroot)
    vox1, vox1_dev = wav_list_reader(dataroot)
    vox_duration = multiprocessing.Manager().list()
    # spk_utt_duration = multiprocessing.Manager().dict()

    share_lock = multiprocessing.Manager().Lock()

    for i in range(len(vox1)):
        queue.put(vox1[i])

    #check_from_queue(queue, que_queue, 1)

    process_list = []
    for i in range(15):
        # pro = Process(target=check_from_queue, args=(queue, que_queue, spk_utt_duration, i, share_lock))
        pro = Process(target=add_duration_vox, args=(queue, que_queue, vox_duration, i, share_lock))
        process_list.append(pro)

    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

    # print(dict(spk_utt_duration))
    # np.save(dataroot+'/spk_utt_duration.npy', dict(spk_utt_duration))
    np.save(dataroot + '/vox_duration.npy', list(vox_duration))
    if que_queue.empty():
        print('\nChecking Fbank features success without error!.')
    else:
        print('Error Fbank features are :')
        while not que_queue.empty():
            ti = que_queue.get()
            print(ti)

    exit(1)

