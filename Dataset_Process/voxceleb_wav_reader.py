import os
from glob import glob
import pathlib
import numpy as np
import sys
import re

np.set_printoptions(threshold=sys.maxsize)


voxceleb_dir = 'dataset'
data_uem = 'dataset/voxceleb1.{subset}.uem'
data_mdtm = 'dataset/voxceleb1.{subset}.mdtm'

list_txt = '{voxceleb_dir}/list.txt'.format(voxceleb_dir=voxceleb_dir)
glob_exp = '{voxceleb_dir}/voxceleb1_txt/*/*.txt'.format(voxceleb_dir=voxceleb_dir)

def check_mk_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def if_load_npy(data_path, npy_path):
    """
    Load npy features or wav from existed files.
    :param data_path: data file's path
    :param npy_path:
    :return:
    """
    if os.path.isfile(npy_path):
        return(np.load(npy_path, allow_pickle=True))
    else:
        dataset = read_extract_audio(data_path)
        np.save(npy_path, dataset)
        return(dataset)

def parse_txt(txt):
    lines = [line.strip() for line in open(txt, 'r').readlines()]
    speaker = lines[0].split('\t')[-1]
    uri = lines[1].split('\t')[-1]
    duration = float(lines[2].split('\t')[-1].split()[0])
    subset = lines[3].split('\t')[-1]

    file_list = []
    for line in lines[5:]:
        file_location, start, end = line.split()
        file_list.append(file_location)


    return subset, uri, speaker, file_list



def find_files(directory, pattern='*/*/*/*.wav'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)


def read_voxceleb_structure(directory):
    voxceleb = []

    #for path_txt in tqdm(glob(glob_exp)):
    for path_txt in glob(glob_exp):
        subset, uri, speaker, file_list = parse_txt(path_txt)

        for file in file_list:
            voxceleb.append({'filename': file, 'speaker_id': speaker, 'uri': uri, 'subset': subset})

    #voxceleb = pd.DataFrame(filelist)
    num_speakers = len(set([datum['speaker_id'] for datum in voxceleb]))
    print('Found {} files with {} different speakers.'.format(str(len(voxceleb)).zfill(7), str(num_speakers).zfill(5)))
    #print(voxceleb.head(10))
    return voxceleb

def read_my_voxceleb_structure(directory):
    voxceleb = []

    data_root = pathlib.Path(directory)
    data_root.cwd()
    print('>>Data root is %s' % str(data_root))
    all_wav_path = list(data_root.glob('*/*/*/*/*.npy'))
    # print(str(pathlib.Path.relative_to(all_wav_path[0], all_wav_path[0].parents[4])).rstrip('.wav'))
    all_wav_path = [str(pathlib.Path.relative_to(path, path.parents[4])).rstrip('.npy') for path in all_wav_path]
    subset = ['dev' if pathlib.Path(path).parent.parent.parent.parent.name=='vox1_dev_wav' else 'test' for path in all_wav_path]
    speaker = [pathlib.Path(path).parent.parent.name for path in all_wav_path]
    all_wav = np.transpose([all_wav_path, subset, speaker])
    for file in all_wav:
        voxceleb.append({'filename': file[0], 'speaker_id': file[2], 'uri': 0, 'subset': file[1]})
        # print(str(file[0]))
        # exit()
    num_speakers = len(set([datum['speaker_id'] for datum in voxceleb]))
    print('>>Found {} files with {} different speakers.'.format(str(len(voxceleb)), str(num_speakers)))
    #print(voxceleb.head(10))
    return voxceleb

def read_extract_audio(directory):
    audio_set = []

    print('>>Data root is %s' % str(directory))

    all_wav_path = []
    for dirs, dirnames, files in os.walk(directory):
        for file in files:
            # print(str(file))
            if re.match(r'^[^\.].*\.npy', str(file)) is not None:
                all_wav_path.append(str(os.path.join(dirs, file)))

    for file in all_wav_path:
        spkid = pathlib.Path(file).parent.parent.name
        audio_set.append({'filename': file.rstrip('.npy'), 'utt_id': file.rstrip('.npy'), 'uri': 0, 'subset': spkid if spkid!='Data' else 'test'})

    print('>>Found {} wav/npy files for extracting xvectors.'.format(len(audio_set)))
    #print(voxceleb.head(10))
    return audio_set

def wav_list_reader(data_path):
    """
        Check if resume dataset variables from local list(.npy).
    :param data_path: the dataset root
    :return: the data list
    """
    voxceleb_list = "Data/voxceleb.npy"
    voxceleb_dev_list = "Data/voxceleb_dev.npy"

    if os.path.isfile(voxceleb_list):
        voxceleb = np.load(voxceleb_list, allow_pickle=True)
        if len(voxceleb)!=153516:
            raise ValueError("The number of wav files may be wrong!")
    else:
        voxceleb = read_my_voxceleb_structure(data_path)
        np.save(voxceleb_list, voxceleb)

    if os.path.isfile(voxceleb_dev_list):
        voxceleb_dev = np.load(voxceleb_dev_list, allow_pickle=True)
    else:
        voxceleb_dev = [datum for datum in voxceleb if datum['subset'] == 'dev']
        np.save(voxceleb_dev_list, voxceleb_dev)

    return voxceleb, voxceleb_dev


# read_my_voxceleb_structure('/data/voxceleb')




