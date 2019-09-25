#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: make_feats.py
@Time: 2019/9/24 下午8:32
@Overview: make feats for missing speakers dataset using multi threads and processes.
"""
from __future__ import print_function
import os
import pathlib
import pdb
import threading
from multiprocessing import Process, Queue
import time

from Process_Data.audio_processing import make_Fbank, conver_to_wav

missing_spks = ['id06996', 'id07001', 'id07002', 'id07008', 'id07017', 'id07031', 'id07032', 'id07043', 'id07044', 'id07057', 'id07062', 'id07063', 'id07065', 'id07072', 'id07080', 'id07084', 'id07085', 'id07117', 'id07118', 'id07133', 'id07135', 'id07148', 'id07153', 'id07165', 'id07179', 'id07181', 'id07182', 'id07183', 'id07186', 'id07187', 'id07200', 'id07205', 'id07206', 'id07220', 'id07232', 'id07243', 'id07256', 'id07273', 'id07275', 'id07278', 'id07293', 'id07299', 'id07305', 'id07308', 'id07334', 'id07341', 'id07342']
# pdb.set_trace()
num_make = 0

def make_feats_spks(spk_id):

    data_root = pathlib.Path('/home/cca01/work2019/Data/voxceleb2/dev/aac/{}'.format(spk_id))
    data_dir = pathlib.Path('/home/cca01/work2019/Data/voxceleb2/dev/aac')

    # print('\nspeaker is %s' % str(spk_id))

    # all the paths of wav files
    # dev/acc/spk_id/utt_group/wav_id.wav
    all_abs_path = list(data_root.glob('*/*.wav'))
    all_rel_path = [str(pathlib.Path.relative_to(path, data_dir)).rstrip('.wav') for path in all_abs_path]

    num_pro = 0
    for datum in all_rel_path:
        # datum likes 'id08648/Hk5G97l1fR0/00064'
        # Data/Voxceleb1/
        # /data/voxceleb/voxceleb1_wav/
        # pdb.set_trace()
        filename = str(data_dir) + '/' +datum + '.wav'
        write_path = 'Data/Voxceleb2/dev/aac/' + datum + '.npy'

        if os.path.exists(filename):
            make_Fbank(filename=filename, write_path=write_path)
        # convert the audio format for m4a.
        # elif os.path.exists(filename.replace('.wav', '.m4a')):
            # conver_to_wav(filename.replace('.wav', '.m4a'),
            #               write_path=args.dataroot + '/voxceleb2/' + datum['filename'] + '.wav')
            #
            # make_Fbank(filename=filename,
            #            write_path=write_path)

        # print('\rThread: {} \t processed {:2f}% {}/{}.'.format(threadid, num_pro / len(all_rel_path), num_pro, len(all_rel_path)), end='\r')
        num_pro += 1



class MyThread(threading.Thread):
    def __init__(self, spk_ids, threadid):
        super(MyThread, self).__init__()  # 重构run函数必须要写
        self.spk_ids = spk_ids
        self.threadid = threadid

    def run(self):
        global num_make
        for spk_id in self.spk_ids:
            make_feats_spks(spk_id)
            num_make += 1
            print('\t{:4d} of speakers making feats completed!'.format(num_make))

class MyProcess(Process):
    def __init__(self, spk_ids, proid, queue):
        super(MyProcess, self).__init__()  # 重构run函数必须要写
        self.spk_ids = spk_ids
        self.proid = proid
        self.queue = queue

    def run(self):
        # global num_make
        for spk_id in self.spk_ids:
            make_feats_spks(spk_id)
            #num_make += 1
            self.queue.put(spk_id)
            print('\t{:4d} of speakers making feats completed!'.format(self.queue.qsize()))

if __name__ == "__main__":
    num_spk = len(missing_spks)
    trunk = int(num_spk / 4)
    start_time = time.time()

    # threadpool = []
    queue = Queue()
    processpool = []
    for i in range(0, 4):
        j = (i+1)*trunk
        if i==3:
            j=num_spk

        print(i*trunk, j)
        # t = MyThread(missing_spks[i*trunk:j], i)
        # t.start()
        # threadpool.append(t)

        p = MyProcess(missing_spks[i*trunk:j], i, queue)
        p.start()
        processpool.append(p)

    # for t in threadpool:
    #     t.join()

    for p in processpool:
        p.join()

    print('For multi process, average making seconds for {} speakers is {}'.format(num_spk, (time.time() - start_time)/num_spk))

    start_time = time.time()
    for spk in missing_spks:
        make_feats_spks(spk)

    print('For one process, average making seconds for {} speakers is {}'.format(num_spk, (
                time.time() - start_time) / num_spk))

"""
For multi threads, average making seconds for 47 speakers is 4.579958657
For one threads, average making seconds for 47 speakers is 4.11888732301

For multi process, average making seconds for 47 speakers is 1.67094940328
For one process, average making seconds for 47 speakers is 3.64203325738
"""



