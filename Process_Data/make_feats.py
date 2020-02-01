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
import Process_Data.constants as c
from Process_Data.audio_processing import make_Fbank, conver_to_wav, Make_Spect, Make_Fbank
from Process_Data.voxceleb2_wav_reader import voxceleb2_list_reader
from Process_Data.voxceleb_wav_reader import wav_list_reader

NUM_JOB = 10
spks = ['id00518']
utts = []

dataset_dir = '/data/voxceleb/voxceleb1_wav'
# dataset_dir = '/home/cca01/work2019/Data/voxceleb1/'
# voxceleb, voxceleb_dev = voxceleb2_list_reader(dataset_dir)
# spks = list(set([datum['speaker_id'] for datum in voxceleb_dev]))
# pdb.set_trace()
vox1, vox1_dev = wav_list_reader(dataset_dir)

utts = [i['filename'] for i in vox1]

# vox1_test = [datum for datum in vox1 if datum['subset'] == 'test']
# spks = list(set([datum['speaker_id'] for datum in vox1]))

data_dir = 'Data/dataset/voxceleb1/fbank24'
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
        # Data/voxceleb1/
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


def make_with_path(path):

    wav_path = dataset_dir + '/{}.wav'.format(path.decode('utf-8'))
    write_path = data_dir + '/{}.npy'.format(path.decode('utf-8'))

    if os.path.exists(write_path):
        try:
            np.load(write_path)
            return
        except Exception:
            print("Error load exsit npy file")

    if os.path.exists(wav_path):
        # np_spec = Make_Spect(wav_path=wav_path, windowsize=0.02, stride=0.01)
        np_fbank = Make_Fbank(filename=wav_path, use_energy=True, nfilt=c.TDNN_FBANK_FILTER)

        file_path = pathlib.Path(write_path)
        if not file_path.parent.exists():
            os.makedirs(str(file_path.parent))

        # np.save(write_path, np_spec)
        np.save(write_path, np_fbank)

    else:
        raise ValueError(str(wav_path) + ' doesn\'t exist.')


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
    def __init__(self, item, proid, queue):
        super(MakeFeatsProcess, self).__init__()  # 重构run函数必须要写
        self.item = item
        self.proid = proid
        self.queue = queue

    def run(self):
        # global num_make
        for i in self.item:
            make_with_path(i)
            self.queue.put(i)

            print('\tProcess {:2d}: {:4d} of items making feats completed!'.format(self.proid, self.queue.qsize()), end='\n')
        print('>>Process {} finished!'.format(self.proid))

if __name__ == "__main__":
    # num_spk = len(spks)
    num_utt = len(utts)

    chunk = int(num_utt / NUM_JOB)
    start_time = time.time()

    # threadpool = []
    queue = Queue()
    processpool = []

    print('make feats for {} utterances.'.format(num_utt))
    for i in range(0, NUM_JOB):
        j = (i+1)*chunk

        if i==(NUM_JOB-1):
            j = num_utt

        # t = MyThread(missing_spks[i*trunk:j], i)
        # t.start()
        # threadpool.append(t)

        p = MakeFeatsProcess(utts[i*chunk:j], i, queue)
        p.start()
        processpool.append(p)

    # for t in threadpool:
    #     t.join()

    for p in processpool:
        p.join()

    print('For multi process, average making seconds for {} speakers is {}'.format(num_utt, (time.time() - start_time)/num_utt))

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



