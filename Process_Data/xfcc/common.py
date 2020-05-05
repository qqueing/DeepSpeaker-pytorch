#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: common.py
@Time: 2020/3/31 8:58 PM
@Overview:
"""
import numpy as np
# from speechpy.functions import frequency_to_mel, mel_to_frequency, triangle
# import soundfile as sf
# import matplotlib.pyplot as plt
from python_speech_features import sigproc, hz2mel, mel2hz, lifter
from scipy import interpolate
from scipy.fft import dct

import Process_Data.constants as c


def hz2amel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + (8000 - hz) / 700.)


def amel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 8000 - 700 * (10 ** (mel / 2595.0) - 1)

def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0,
                    highfreq=None, filtertype='mel', multi_weight=False):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """

    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    if filtertype == 'mel':
        # compute points evenly spaced in mels
        lowmel = hz2mel(lowfreq)
        highmel = hz2mel(highfreq)
        melpoints = np.linspace(lowmel, highmel, nfilt + 2)
        # our points are in Hz, but we use fft bins, so we have to convert from Hz to fft bin number
        bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)
    elif filtertype == 'amel':
        # compute points evenly spaced in mels
        lowmel = hz2amel(lowfreq)
        highmel = hz2amel(highfreq)
        melpoints = np.linspace(lowmel, highmel, nfilt + 2)
        # our points are in Hz, but we use fft bins, so we have to convert from Hz to fft bin number
        bin = np.floor((nfft + 1) * amel2hz(melpoints) / samplerate)

    elif filtertype == 'linear':
        linearpoints = np.linspace(lowfreq, highfreq, nfilt + 2)
        # our points are in Hz, but we use fft bins, so we have to convert from Hz to fft bin number
        bin = np.floor((nfft + 1) * linearpoints / samplerate)

    elif filtertype.startswith('dnn'):
        x = np.arange(0, 161) * samplerate / 2 / 160
        if filtertype.endswith('timit.fix'):
            y = np.array(c.TIMIT_FIlTER_FIX)
        elif filtertype.endswith('timit.var'):
            y = np.array(c.TIMIT_FIlTER_VAR)
        elif filtertype.endswith('timit.mdv'):
            y = np.array(c.TIMIT_FIlTER_MDV)
        elif filtertype.endswith('libri.fix'):
            y = np.array(c.LIBRI_FILTER_FIX)
        elif filtertype.endswith('libri.var'):
            y = np.array(c.LIBRI_FILTER_VAR)
        elif filtertype.endswith('vox1.soft'):
            y = np.array(c.VOX_FILTER_SOFT)
        elif filtertype == 'dnn.vox1':
            y = np.array(c.VOX_FILTER)

        f = interpolate.interp1d(x, y)
        x_new = np.arange(nfft // 2 + 1) * samplerate / 2 / (nfft // 2)
        lowfreq_idx = np.where(x_new >= lowfreq)[0]
        highfreq_idx = np.where(x_new <= highfreq)[0]
        ynew = f(x_new)  # 计算插值结果

        ynew[:int(lowfreq_idx[0])] = 0
        if highfreq_idx[-1] < len(x_new) - 1:
            ynew[int(highfreq[-1] + 1):] = 0

        weight = ynew / np.sum(ynew)

        bin = []
        bin.append(lowfreq_idx[0])

        for j in range(nfilt):
            num_wei = 0.
            for i in range(nfft // 2 + 1):
                num_wei += weight[i]
                if num_wei > (j + 1) / (nfilt + 1):
                    bin.append(i - 1)
                    break
                else:
                    continue

        bin.append(highfreq_idx[-1])

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])

        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])

    if multi_weight:
        y = np.array(c.TIMIT_FIlTER_VAR)
        fbank = fbank * (y / y.max())

    return fbank


def local_fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
                nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                filtertype='mel', winfunc=lambda x: np.hamming((x,)),
                multi_weight=False):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq = highfreq or samplerate / 2
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = sigproc.powspec(frames, nfft)
    energy = np.sum(pspec, 1)  # this stores the total energy in each frame
    energy = np.where(energy == 0, np.finfo(float).eps, energy)  # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq, filtertype, multi_weight=multi_weight)
    feat = np.dot(pspec, fb.T)  # compute the filterbank energies
    feat = np.where(feat == 0, np.finfo(float).eps, feat)  # if feat is zero, we get problems with log

    # feat = np.log1p(feat)

    return feat, energy


def local_mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
               nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
               ceplifter=22, filtertype='mel',
               appendEnergy=True, winfunc=lambda x: np.hamming((x,))):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    feat, energy = local_fbank(signal=signal, samplerate=samplerate, winlen=winlen,
                               winstep=winstep, nfilt=nfilt, nfft=nfft,
                               lowfreq=lowfreq, highfreq=highfreq, preemph=preemph,
                               winfunc=winfunc, filtertype=filtertype)
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    feat = lifter(feat, ceplifter)
    if appendEnergy: feat[:, 0] = np.log(energy)  # replace first cepstral coefficient with log of frame energy
    return feat

# filters = get_filterbanks(nfilt=24, nfft=512, samplerate=16000, lowfreq=0, highfreq=None, filtertype='dnn')
# plt.show()

# wav_path = 'Data/dataset/voxceleb1/vox1_dev_wav/wav/id10001/1zcIwhmdeo4/00001.wav'
# from scipy.io import wavfile
#
# # samples, samplerate = sf.read(wav_path)
# samplerate, samples = wavfile.read(wav_path)
# feat, energy = fbank(samples, samplerate, nfilt=64, filtertype='mel')
# log_fb = np.log(feat)
# a = log_fb[0]
# a = (a-np.mean(a))/np.std(a)
# plt.plot(a)
# plt.show()
#
# feat, energy = fbank(samples, samplerate, nfilt=64, filtertype='linear')
# log_fb = np.log(feat)
# a = log_fb[0]
# a = (a-np.mean(a))/np.std(a)
# plt.plot(a)
# plt.show()
#
# feat, energy = fbank(samples, samplerate, nfilt=64, filtertype='dnn')
# log_fb = np.log(feat)
# a = log_fb[0]
# a = (a-np.mean(a))/np.std(a)
# plt.plot(a)
# plt.show()
#
# kaldi_out = [19.883001,  8.636156,  9.324322, 14.557433, 15.946762, 16.232332,
#        15.824728, 16.404913, 18.172726, 17.71976 , 15.660718, 17.280998,
#        17.957315, 17.496141, 17.734505, 16.460432, 17.788591, 17.868443,
#        16.362251, 16.308823, 17.069942, 18.959517, 18.530521, 16.378983,
#        15.806672, 16.423117, 16.776777, 16.547352, 16.32962 , 19.124495,
#        19.058388, 18.82699 , 18.509348, 19.193333, 18.395838, 18.070152,
#        18.014786, 17.284107, 17.089983, 17.378532, 18.002579, 16.869461,
#        17.973913, 18.372267, 18.360933, 19.562916, 21.65837 , 20.915466,
#        20.076878, 20.36837 , 20.000563, 19.351028, 19.47047 , 19.580456,
#        19.721878, 19.557446, 20.073544, 18.958134, 18.773273, 18.282299,
#        18.54605 , 19.301865, 18.725395, 19.297384, 18.555035]
# a = np.array(kaldi_out)
# a = (a-np.mean(a))/np.std(a)
# plt.plot(a)
# plt.show()
