import numpy as np
from python_speech_features import fbank, delta

from Process_Data import constants as c
import torch
import librosa

from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment

import os
import pathlib
import pdb


def mk_MFB(filename, sample_rate=c.SAMPLE_RATE, use_delta=c.USE_DELTA, use_scale=c.USE_SCALE, use_logscale=c.USE_LOGSCALE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    #audio = audio.flatten()

    filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=c.FILTER_BANK, winlen=0.025)

    if use_logscale:
        filter_banks = 20 * np.log10(np.maximum(filter_banks, 1e-5))

    if use_delta:
        delta_1 = delta(filter_banks, N=1)
        delta_2 = delta(delta_1, N=1)

        filter_banks = normalize_frames(filter_banks, Scale=use_scale)
        delta_1 = normalize_frames(delta_1, Scale=use_scale)
        delta_2 = normalize_frames(delta_2, Scale=use_scale)

        frames_features = np.hstack([filter_banks, delta_1, delta_2])
    else:
        filter_banks = normalize_frames(filter_banks, Scale=use_scale)
        frames_features = filter_banks

    np.save(filename.replace('.wav', '.npy'), frames_features)

    return

def make_Fbank(filename,
               write_path, # sample_rate=c.SAMPLE_RATE,
               use_delta=c.USE_DELTA,
               use_scale=c.USE_SCALE,
               nfilt=c.FILTER_BANK,
               use_logscale=c.USE_LOGSCALE,
               use_energy=c.USE_ENERGY,
               normalize=c.NORMALIZE):

    if not os.path.exists(filename):
        raise ValueError('wav file does not exist.')

    sample_rate, audio = wavfile.read(filename)
    # audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    #audio = audio.flatten()

    filter_banks, energies = fbank(audio,
                                   samplerate=sample_rate,
                                   nfilt=nfilt,
                                   winlen=0.025,
                                   winfunc=np.hamming)

    if use_energy:
        energies = energies.reshape(energies.shape[0], 1)
        filter_banks = np.concatenate((energies, filter_banks), axis=1)
        # frames_features[:, 0] = np.log(energies)

    if use_logscale:
        # filter_banks = 20 * np.log10(np.maximum(filter_banks, 1e-5))
        filter_banks = np.log(np.maximum(filter_banks, 1e-5))

    # Todo: extract the normalize step?
    if use_delta:
        delta_1 = delta(filter_banks, N=1)
        delta_2 = delta(delta_1, N=1)

        filter_banks = normalize_frames(filter_banks, Scale=use_scale)
        delta_1 = normalize_frames(delta_1, Scale=use_scale)
        delta_2 = normalize_frames(delta_2, Scale=use_scale)

        frames_features = np.hstack([filter_banks, delta_1, delta_2])

    if normalize:
        filter_banks = normalize_frames(filter_banks, Scale=use_scale)

    frames_features = filter_banks

    file_path = pathlib.Path(write_path)
    if not file_path.parent.exists():
        os.makedirs(str(file_path.parent))

    np.save(write_path, frames_features)

    # np.save(filename.replace('.wav', '.npy'), frames_features)
    return

def GenerateSpect(wav_path, write_path, windowsize=25, stride=10, nfft=c.NUM_FFT):
    """
    Pre-computing spectrograms for wav files
    :param wav_path: path of the wav file
    :param write_path: where to write the spectrogram .npy file
    :param windowsize:
    :param stride:
    :param nfft:
    :return: None
    """
    if not os.path.exists(wav_path):
        raise ValueError('wav file does not exist.')
    #pdb.set_trace()

    sample_rate, samples = wavfile.read(wav_path)
    sample_rate_norm = int(sample_rate / 1e3)
    frequencies, times, spectrogram = signal.spectrogram(x=samples, fs=sample_rate, window=signal.hamming(windowsize * sample_rate_norm), noverlap=(windowsize-stride) * sample_rate_norm, nfft=nfft)

    # Todo: store the whole spectrogram
    # spectrogram = spectrogram[:, :300]
    # while spectrogram.shape[1]<300:
    #     # Copy padding
    #     spectrogram = np.concatenate((spectrogram, spectrogram), axis=1)
    #
    #     # raise ValueError("The dimension of spectrogram is less than 300")
    # spectrogram = spectrogram[:, :300]
    # maxCol = np.max(spectrogram,axis=0)
    # spectrogram = np.nan_to_num(spectrogram / maxCol)
    # spectrogram = spectrogram * 255
    # spectrogram = spectrogram.astype(np.uint8)

    # For voxceleb1
    # file_path = wav_path.replace('Data/Voxceleb1', 'Data/voxceleb1')
    # file_path = file_path.replace('.wav', '.npy')

    file_path = pathlib.Path(write_path)
    if not file_path.parent.exists():
        os.makedirs(str(file_path.parent))

    np.save(write_path, spectrogram)

    # return spectrogram

def conver_to_wav(filename, write_path, format='m4a'):
    """
    Convert other formats into wav.
    :param filename: file path for the audio.
    :param write_path:
    :param format: formats that ffmpeg supports.
    :return: None. write the wav to local.
    """
    if not os.path.exists(filename):
        raise ValueError('File may not exist.')

    if not pathlib.Path(write_path).parent.exists():
        os.makedirs(str(pathlib.Path(write_path).parent))

    sound = AudioSegment.from_file(filename, format=format)
    sound.export(write_path, format="wav")

def read_MFB(filename):
    #audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    #audio = audio.flatten()
    try:
        audio = np.load(filename.replace('.wav', '.npy'))
    except Exception:

        raise ValueError("Load {} error!".format(filename))

    return audio

def read_from_npy(filename):
    """
    read features from npy files
    :param filename: the path of wav files.
    :return:
    """
    #audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    #audio = audio.flatten()
    audio = np.load(filename.replace('.wav', '.npy'))

    return audio

class truncatedinputfromMFB(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, input_per_file=1):

        super(truncatedinputfromMFB, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):

        network_inputs = []
        num_frames = len(frames_features)
        import random

        for i in range(self.input_per_file):

            j = random.randrange(c.NUM_PREVIOUS_FRAME, num_frames - c.NUM_NEXT_FRAME)
            if not j:
                frames_slice = np.zeros(c.NUM_FRAMES, c.FILTER_BANK, 'float64')
                frames_slice[0:(frames_features.shape)[0]] = frames_features.shape
            else:
                frames_slice = frames_features[j - c.NUM_PREVIOUS_FRAME:j + c.NUM_NEXT_FRAME]
            network_inputs.append(frames_slice)

        return np.array(network_inputs)

class concateinputfromMFB(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, input_per_file=1):

        super(concateinputfromMFB, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):

        network_inputs = []
        num_frames = len(frames_features)

        import math
        # pdb.set_trace()
        num_utt = math.ceil(float(num_frames) / c.NUM_FRAMES)
        output = np.zeros((int(num_utt*c.NUM_FRAMES)-num_frames, frames_features.shape[1]))
        output = np.concatenate((frames_features, output), axis=0)

        for i in range(int(num_utt)):
            frames_slice = output[i*c.NUM_FRAMES:(i+1)*c.NUM_FRAMES]
            network_inputs.append(frames_slice)
        # pdb.set_trace()
        network_inputs = np.array(network_inputs)
        #return_output = network_inputs.reshape((1, network_inputs.shape[0], network_inputs.shape[1], network_inputs.shape[2]))

        return network_inputs

class truncatedinputfromSpectrogram(object):
    """truncated input from Spectrogram
    """
    def __init__(self, input_per_file=1):

        super(truncatedinputfromSpectrogram, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):

        network_inputs = []
        frames_features = np.swapaxes(frames_features, 0, 1)
        num_frames = len(frames_features)
        import random

        for i in range(self.input_per_file):

            j=0

            if c.NUM_PREVIOUS_FRAME_SPECT <= (num_frames - c.NUM_NEXT_FRAME_SPECT):
                j = random.randrange(c.NUM_PREVIOUS_FRAME_SPECT, num_frames - c.NUM_NEXT_FRAME_SPECT)

            #j = random.randrange(c.NUM_PREVIOUS_FRAME_SPECT, num_frames - c.NUM_NEXT_FRAME_SPECT)
            # If len(frames_features)<NUM__FRAME_SPECT, then apply zero padding.
            if j==0:
                frames_slice = np.zeros((c.NUM_FRAMES_SPECT, c.NUM_FFT/2+1), dtype=np.float32)
                frames_slice[0:(frames_features.shape[0])] = frames_features
            else:
                frames_slice = frames_features[j - c.NUM_PREVIOUS_FRAME_SPECT:j + c.NUM_NEXT_FRAME_SPECT]

            network_inputs.append(frames_slice)

        return np.array(network_inputs)


def read_audio(filename, sample_rate=c.SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio

#this is not good
#def normalize_frames(m):
#    return [(v - np.mean(v)) / (np.std(v) + 2e-12) for v in m]

def normalize_frames(m, Scale=True):
    """
    Normalize frames with mean and variance
    :param m:
    :param Scale:
    :return:
    """
    if Scale:
        return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))


def pre_process_inputs(signal=np.random.uniform(size=32000), target_sample_rate=8000,use_delta = c.USE_DELTA):
    filter_banks, energies = fbank(signal, samplerate=target_sample_rate, nfilt=c.FILTER_BANK, winlen=0.025)
    delta_1 = delta(filter_banks, N=1)
    delta_2 = delta(delta_1, N=1)

    filter_banks = normalize_frames(filter_banks)
    delta_1 = normalize_frames(delta_1)
    delta_2 = normalize_frames(delta_2)

    if use_delta:
        frames_features = np.hstack([filter_banks, delta_1, delta_2])
    else:
        frames_features = filter_banks
    num_frames = len(frames_features)
    network_inputs = []
    """Too complicated
    for j in range(c.NUM_PREVIOUS_FRAME, num_frames - c.NUM_NEXT_FRAME):
        frames_slice = frames_features[j - c.NUM_PREVIOUS_FRAME:j + c.NUM_NEXT_FRAME]
        #network_inputs.append(np.reshape(frames_slice, (32, 20, 3)))
        network_inputs.append(frames_slice)
        
    """
    import random
    j = random.randrange(c.NUM_PREVIOUS_FRAME, num_frames - c.NUM_NEXT_FRAME)
    frames_slice = frames_features[j - c.NUM_PREVIOUS_FRAME:j + c.NUM_NEXT_FRAME]
    network_inputs.append(frames_slice)
    return np.array(network_inputs)

class truncatedinput(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __call__(self, input):

        #min_existing_frames = min(self.libri_batch['raw_audio'].apply(lambda x: len(x)).values)
        want_size = int(c.TRUNCATE_SOUND_FIRST_SECONDS * c.SAMPLE_RATE)
        if want_size > len(input):
            output = np.zeros((want_size,))
            output[0:len(input)] = input
            #print("biho check")
            return output
        else:
            return input[0:want_size]


class toMFB(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __call__(self, input):

        output = pre_process_inputs(input, target_sample_rate=c.SAMPLE_RATE)
        return output


class totensor(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            #img = torch.from_numpy(pic.transpose((0, 2, 1)))
            #return img.float()
            # pdb.set_trace()
            img = torch.FloatTensor(pic.transpose((0, 2, 1)))
            #img = np.float32(pic.transpose((0, 2, 1)))
            return img

            #img = torch.from_numpy(pic)
            # backward compatibility

class to4tensor(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((0, 2, 1)))
            #return img.float()
            # pdb.set_trace()
            #img = torch.FloatTensor(pic.transpose((1, 0, 3, 2)))
            #img = np.float32(pic.transpose((0, 2, 1)))
            return img.unsqueeze(0)
            #img = torch.from_numpy(pic)
            # backward compatibility

class tonormal(object):


    def __init__(self):
        self.mean = 0.013987
        self.var = 1.008

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient

        print(self.mean)
        self.mean+=1
        #for t, m, s in zip(tensor, self.mean, self.std):
        #    t.sub_(m).div_(s)
        return tensor
