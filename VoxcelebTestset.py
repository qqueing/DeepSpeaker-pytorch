import os
import numpy as np
import torch.utils.data as data





def get_lfw_paths(pairs_path,lfw_dir,file_ext="wav"):

    def read_lfw_pairs(pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            # for line in f.readlines()[1:]:
            for line in f.readlines():
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def read_lfw_pairs2(pairs_filename):

        pairs = [line.strip().split() for line in open(pairs_filename, 'r').readlines()]
        return np.array(pairs)

    #pairs = read_lfw_pairs2(pairs_path)
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
        path0 = lfw_dir +'/voxceleb1_wav/' + pair[1]
        path1 = lfw_dir +'/voxceleb1_wav/' + pair[2]


        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list.append((path0,path1,issame))
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list

class VoxcelebTestset(data.Dataset):
    '''
    '''
    def __init__(self,  dir,pairs_path, loader, transform=None):

        #super(VoxcelebTestset, self).__init__(dir,transform)


        self.pairs_path = pairs_path
        self.loader = loader
        # LFW dir contains 2 folders: faces and lists
        self.validation_images = get_lfw_paths(self.pairs_path,dir)
        self.transform = transform




    def __getitem__(self, index):
        '''

        Args:
            index: Index of the triplet or the matches - not of a single image

        Returns:

        '''

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            return self.transform(img)

        (path_1,path_2,issame) = self.validation_images[index]
        img1, img2 = transform(path_1), transform(path_2)
        return img1, img2, issame


    def __len__(self):
        return len(self.validation_images)
