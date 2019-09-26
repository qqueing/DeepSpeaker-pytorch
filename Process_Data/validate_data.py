#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: validate_data.py
@Time: 2019/9/26 上午10:51
@Overview:
"""
from __future__ import print_function

import os
from multiprocessing import Queue, Process

import numpy as np
from Process_Data.voxceleb2_wav_reader import voxceleb2_list_reader

dataroot = '/home/cca01/work2019/yangwenhao/mydataset/voxceleb2/fbank64'

num_pro = 0.
skip_wav = 0.


def check_feats(vox2):
    #for datum in voxceleb2:
    # Data/Voxceleb1/
    # /data/voxceleb/voxceleb1_wav/
    # pdb.set_trace()
    #filename = '/home/cca01/work2019/Data/voxceleb2/' + datum['filename'] + '.wav'
    write_path = dataroot + '/' + datum['filename'] + '.npy'

    try:
        item = np.load(write_path)

        if item.shape[1]!=64:
            raise ValueError('feature {} shape error!\n'.format(write_path))

    except ValueError:
        raise ValueError('file {} has error!\n'.format(write_path))

    except Exception:
        raise Exception('Load \'{}\' npy file error!\n'.format(write_path))

def check_from_queue(queue, cpid):

    while not queue.empty():
        vox = queue.get()
        check_feats(vox)
        print('Process {}: There are {:8d} features left.'.format(cpid, queue.qsize()), end='\r')

if __name__ == '__main__':
    queue = Queue()
    voxceleb2, voxceleb2_dev = voxceleb2_list_reader(dataroot)

    for datum in voxceleb2_dev:
        queue.put(datum)

    pro1 = Process(target=check_from_queue, args=(queue, 1))
    pro2 = Process(target=check_from_queue, args=(queue, 2))
    pro3 = Process(target=check_from_queue, args=(queue, 3))
    pro4 = Process(target=check_from_queue, args=(queue, 4))

    pro1.start()
    pro2.start()
    pro3.start()
    pro4.start()

    #print(queue.get())
    pro1.join()
    pro2.join()
    pro3.join()
    pro4.join()

    print('\nChecking Fbank features success without error!.')
    exit(1)

