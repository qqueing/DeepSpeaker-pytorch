import numpy as np
from python_speech_features import fbank, delta

import constants as c

import librosa


def mk_MFB(filename, sample_rate=c.SAMPLE_RATE,use_delta = c.USE_DELTA,use_scale = c.USE_SCALE,use_logscale = c.USE_LOGSCALE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    #audio = audio.flatten()


    filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=c.FILTER_BANK, winlen=0.025)

    if use_logscale:
        filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))

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



    np.save(filename.replace('.wav', '.npy'),frames_features)

    return

def read_MFB(filename):
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




def read_audio(filename, sample_rate=c.SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio

#this is not good
#def normalize_frames(m):
#    return [(v - np.mean(v)) / (np.std(v) + 2e-12) for v in m]

def normalize_frames(m,Scale=True):
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
import torch
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
            img = torch.FloatTensor(pic.transpose((0, 2, 1)))
            #img = np.float32(pic.transpose((0, 2, 1)))
            return img

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
