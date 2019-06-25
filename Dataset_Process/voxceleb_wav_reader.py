import os
from glob import glob
import pathlib
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


voxceleb_dir = 'dataset'
data_uem = 'dataset/voxceleb1.{subset}.uem'
data_mdtm = 'dataset/voxceleb1.{subset}.mdtm'

list_txt = '{voxceleb_dir}/list.txt'.format(voxceleb_dir=voxceleb_dir)
glob_exp = '{voxceleb_dir}/voxceleb1_txt/*/*.txt'.format(voxceleb_dir=voxceleb_dir)


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

# read_my_voxceleb_structure('/data/voxceleb')




