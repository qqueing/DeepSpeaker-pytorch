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

import numpy as np
from Process_Data.voxceleb2_wav_reader import voxceleb2_list_reader
from Process_Data.voxceleb_wav_reader import wav_list_reader

dataroot = '/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/Fbank64_Norm'

num_pro = 0.
skip_wav = 0.


def check_from_queue(queue, que_queue, cpid):
    # return np.load('/home/cca01/work2019/yangwenhao/mydataset/voxceleb2/fbank64/dev/aac/id02912/88v-sPZl5-w/00010.npy')
    while not queue.empty():
         vox2 = queue.get()

         write_path = dataroot + '/' + vox2['filename'].decode('utf-8') + '.npy'
         # pdb.set_trace()
         try:
             item = np.load(write_path)
             # print('')
         except Exception:
             que_queue.put(vox2)
             # break

         print('\rProcess {}: There are {:8d} features left.'.format(cpid, queue.qsize()), end='\r')

if __name__ == '__main__':
    queue = Queue()
    que_queue = Queue()
    # voxceleb2, voxceleb2_dev = voxceleb2_list_reader(dataroot)
    vox1, vox1_dev = wav_list_reader(dataroot)

    for datum in vox1:
        queue.put(datum)

    #check_from_queue(queue, que_queue, 1)
    pro1 = Process(target=check_from_queue, args=(queue, que_queue, 1))
    pro2 = Process(target=check_from_queue, args=(queue, que_queue, 2))
    pro3 = Process(target=check_from_queue, args=(queue, que_queue, 3))
    pro4 = Process(target=check_from_queue, args=(queue, que_queue, 4))

    pro1.start()
    pro2.start()
    pro3.start()
    pro4.start()

    #print(queue.get())
    pro1.join()
    pro2.join()
    pro3.join()
    pro4.join()
    if que_queue.empty():
        print('\nChecking Fbank features success without error!.')
    else:
        print('Error Fbank features are :')
        while not que_queue.empty():
            ti = que_queue.get()
            print(ti)

    exit(1)

