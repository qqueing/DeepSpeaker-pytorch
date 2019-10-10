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
import numpy as np

from Process_Data.audio_processing import make_Fbank, conver_to_wav
from Process_Data.voxceleb2_wav_reader import voxceleb2_list_reader
from Process_Data.voxceleb_wav_reader import wav_list_reader

NUM_JOB = 4
spks = ['id00518']

# dataset_dir = '/data/voxceleb/voxceleb1_wav'
dataset_dir = '/home/cca01/work2019/Data/voxceleb2/'
# voxceleb, voxceleb_dev = voxceleb2_list_reader(dataset_dir)
# spks = list(set([datum['speaker_id'] for datum in voxceleb_dev]))

# vox1, vox1_dev = wav_list_reader(dataset_dir)
# vox1_test = [datum for datum in vox1 if datum['subset'] == 'test']
# spks = list(set([datum['speaker_id'] for datum in vox1_test]))

data_dir = '/home/cca01/work2019/yangwenhao/mydataset/voxceleb2/fbank24'
dataset_path = pathlib.Path(dataset_dir)
# dev/aac/{}'
# data_dir = pathlib.Path('/home/cca01/')

# pdb.set_trace()
num_make = 0

def make_feats_spks(spk_id):
    # print('\nspeaker is %s' % str(spk_id))
    spkid_path = pathlib.Path(dataset_dir + '/dev/aac/{}'.format(spk_id))

    # all the paths of wav files
    # dev/acc/spk_id/utt_group/wav_id.wav
    all_wav = list(spkid_path.glob('*/*.wav'))

    num_pro = 0.
    skip_wav = 0.
    for datum in all_wav:
        # Data/Voxceleb1/
        # /data/voxceleb/voxceleb1_wav/
        # pdb.set_trace()
        filename = str(datum)
        write_path = data_dir + '/' + str(datum.relative_to(dataset_path)).replace('.wav', '.npy')

        if os.path.exists(write_path):
            try:
                np.load(write_path)
                # print('')
            except Exception:
                print("Error load exsit npy file")

            num_pro += 1
            skip_wav += 1
            continue

        if os.path.exists(filename):
            make_Fbank(filename=filename,
                       write_path=write_path,
                       nfilt=23,
                       use_energy=True
                       )
            num_pro += 1
        # convert the audio format for m4a.
        # elif os.path.exists(filename.replace('.wav', '.m4a')):
        #     conver_to_wav(filename.replace('.wav', '.m4a'),
        #                   write_path=data_dir + '/' + datum['filename'] + '.wav')
        #
        #     make_Fbank(filename=args.dataroot + '/' + datum['filename'] + '.wav',
        #                write_path=write_path,
        #                nfilt=c.TDNN_FBANK_FILTER,
        #                use_energy=True)
        #     num_pro += 1
        else:
            raise ValueError(filename + ' doesn\'t exist.')

        # print('\tPreparing for speaker {}.\tProcessed {:2f}% {}/{}.\tSkipped {} wav files.'.format(spk_id,
        #                                                                                            100 * num_pro / len(all_wav),
        #                                                                                            num_pro,
        #                                                                                            len(all_wav),
        #                                                                                            skip_wav), end='\r')

# class MyThread(threading.Thread):
#     def __init__(self, spk_ids, threadid):
#         super(MyThread, self).__init__()  # 重构run函数必须要写
#         self.spk_ids = spk_ids
#         self.threadid = threadid
#
#     def run(self):
#         global num_make
#         for spk_id in self.spk_ids:
#             make_feats_spks(spk_id)
#             num_make += 1
#             print('\t{:4d} of speakers making feats completed!'.format(num_make))


class MakeFeatsProcess(Process):
    def __init__(self, spk_ids, proid, queue):
        super(MakeFeatsProcess, self).__init__()  # 重构run函数必须要写
        self.spk_ids = spk_ids
        self.proid = proid
        self.queue = queue

    def run(self):
        # global num_make
        for spk_id in self.spk_ids:
            make_feats_spks(spk_id)
            #num_make += 1
            self.queue.put(spk_id)
            print('\tProcess {:2d}: {:4d} of speakers making feats completed!'.format(self.proid, self.queue.qsize()), end='\n')
        print('>>Process {} finished!'.format(self.proid))

if __name__ == "__main__":
    num_spk = len(spks)
    trunk = int(num_spk / NUM_JOB)
    start_time = time.time()

    # threadpool = []
    queue = Queue()
    processpool = []
    print('make feats for {} speakers.'.format(num_spk))
    for i in range(0, NUM_JOB):
        j = (i+1)*trunk

        if i==(NUM_JOB-1):
            j = num_spk

        # t = MyThread(missing_spks[i*trunk:j], i)
        # t.start()
        # threadpool.append(t)

        p = MakeFeatsProcess(spks[i*trunk:j], i, queue)
        p.start()
        processpool.append(p)

    # for t in threadpool:
    #     t.join()

    for p in processpool:
        p.join()

    print('For multi process, average making seconds for {} speakers is {}'.format(num_spk, (time.time() - start_time)/num_spk))

    # start_time = time.time()
    # for spk in missing_spks:
    #     make_feats_spks(spk)
    #
    # print('For one process, average making seconds for {} speakers is {}'.format(num_spk, (
    #             time.time() - start_time) / num_spk))

"""
For multi threads, average making seconds for 47 speakers is 4.579958657
For one threads, average making seconds for 47 speakers is 4.11888732301

For multi process, average making seconds for 47 speakers is 1.67094940328
For one process, average making seconds for 47 speakers is 3.64203325738
"""



