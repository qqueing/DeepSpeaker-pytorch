
import os
from glob import glob

import numpy as np

np.set_printoptions(threshold=np.nan)



voxceleb_dir = 'voxceleb'
data_uem = 'data/voxceleb1.{subset}.uem'
data_mdtm = 'data/voxceleb1.{subset}.mdtm'

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


