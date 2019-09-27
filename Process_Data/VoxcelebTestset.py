import os
import pdb

import torch.utils.data as data


def get_test_paths(pairs_path, db_dir, file_ext="wav"):
    print('Verification list file is in: ' + pairs_path)
    print('Verification acoustic feature file is in: ' + db_dir)
    pairs = [line.strip().split() for line in open(pairs_path, 'r').readlines()]
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []

    #pairs = random.sample(pairs, 100)
    #for i in tqdm(range(len(pairs))):
    for pair in pairs:
        #pair = pairs[i]
        if pair[0] == '1':
            issame = True
        else:
            issame = False
        path0 = db_dir +'/vox1_test_wav/wav/' + pair[1]
        path1 = db_dir +'/vox1_test_wav/wav/' + pair[2]

        path0_npy = db_dir +'/vox1_test_wav/wav/' + pair[1].rstrip('.wav') + '.npy'
        path1_npy = db_dir +'/vox1_test_wav/wav/' + pair[2].rstrip('.wav') + '.npy'

        # pdb.set_trace()
        # print(path1_npy)
        if os.path.exists(path0_npy) and os.path.exists(path1_npy):    # Only add the pair if both paths exist
            path_list.append((path0, path1, issame))
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1

    if nrof_skipped_pairs>0:
        print('Skipped %d pairs' % nrof_skipped_pairs)
    print('The number of Verification pairs is %d.' % len(path_list))
    return path_list

class VoxcelebTestset(data.Dataset):
    '''
    '''
    def __init__(self, dir, pairs_path, loader, transform=None):

        self.pairs_path = pairs_path
        self.loader = loader
        self.validation_images = get_test_paths(pairs_path=self.pairs_path, db_dir=dir)
        self.transform = transform


    def __getitem__(self, index):
        '''

        Args:
            index: Index of the triplet or the matches - not of a single features
        Returns:

        '''

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
            """
            img = self.loader(img_path)
            return self.transform(img)

        (path_1,path_2,issame) = self.validation_images[index]
        img1, img2 = transform(path_1), transform(path_2)
        return img1, img2, issame


    def __len__(self):
        return len(self.validation_images)
