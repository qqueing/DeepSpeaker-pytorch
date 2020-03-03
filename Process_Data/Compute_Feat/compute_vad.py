#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: compute_vad.py
@Time: 2019/9/20 上午10:57
@Overview: Implement VAD using python from kaldi.

librosa package load wav data with float32, while in kaldi and scipy.io, it should be int16!!!

"""
import argparse
import numpy as np
# import librosa
# from python_speech_features import fbank, delta, mfcc
import Process_Data.constants as c

# parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# parser.add_argument('--vad-energy-threshold', type=float, default=5.5, metavar='E',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--vad-energy-mean-scale', type=float, default=0.5, metavar='E',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--vad-proportion-threshold', type=float, default=0.12, metavar='E',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--vad-frames-context', type=int, default=2, metavar='E',
#                     help='number of epochs to train (default: 10)')
# opts = parser.parse_args()

def ComputeVadEnergy(feats, output_voiced):
    T = len(feats)
    # output_voiced->Resize(T);

    if (T == 0):
        print("Empty features")
        return

    # column zero is log - energy.
    log_energy = feats[:, 0]

    # CopyColFromMat(feats, 0); // column zero is log-energy.
    # energy_threshold = opts.vad_energy_threshold
    energy_threshold = c.VAD_ENERGY_THRESHOLD

    # if (opts.vad_energy_mean_scale != 0.0):
    #     assert (opts.vad_energy_mean_scale > 0.0)
    #     energy_threshold += opts.vad_energy_mean_scale * np.sum(log_energy) / T
    if (c.VAD_ENERGY_MEAN_SCALE != 0.0):
        assert(c.VAD_ENERGY_MEAN_SCALE > 0.0)
        energy_threshold += c.VAD_ENERGY_MEAN_SCALE * np.sum(log_energy) / T

    # assert (opts.vad_frames_context >= 0);
    # assert (opts.vad_proportion_threshold > 0.0 and opts.vad_proportion_threshold < 1.0)
    assert (c.VAD_FRAMES_CONTEXT >= 0)
    assert (c.VAD_PROPORTION_THRESHOLD > 0.0 and c.VAD_PROPORTION_THRESHOLD < 1.0)

    for t in range(0, T):

        # log_energy_data = log_energy[:][0]
        num_count = 0
        den_count = 0
        # context = opts.vad_frames_context
        context = c.VAD_FRAMES_CONTEXT

        for t2 in range(t - context-1, t + context):
            if (t2 >= 0 and t2 < T):
                den_count+=1

                if (log_energy[t2] > energy_threshold):
                    num_count+=1

        # if (num_count >= den_count * opts.vad_proportion_threshold):
        if (num_count >= den_count * c.VAD_PROPORTION_THRESHOLD):
          output_voiced.append(1.0)
        else:
          output_voiced.append(0.0)

    # return output_voiced


# fbank = np.load('Data/dataset/enroll/id10270/5r0dWxy17C8/00001.npy')
#
# audio, sr = librosa.load('Data/dataset/enroll/id10270/5r0dWxy17C8/00001.wav', sr=16000, mono=True)
#
# from scipy.io import wavfile
# sample_rate, samples = wavfile.read('Data/dataset/enroll/id10270/5r0dWxy17C8/00001.wav')
#
# mfcc1 = mfcc(audio, samplerate=16000, numcep=30, winlen=0.025)
# mfcc2 = mfcc(samples, samplerate=16000, numcep=30, winlen=0.025)
#
# voice1 = []
# ComputeVadEnergy(mfcc1, voice1)
#
# voice2 = []
# ComputeVadEnergy(mfcc2, voice2)

# print(voice)