from __future__ import print_function


import numpy as np

import torch.utils.data as data


def find_classes(voxceleb):
    classes = list(voxceleb['speaker_id'].unique())
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def generate_triplets(imgs, num_triplets,n_classes):
    def create_indices(_imgs):
        inds = dict()
        for idx, (img_path,label) in enumerate(_imgs):
            if label not in inds:
                inds[label] = []
            inds[label].append(img_path)
        return inds

    triplets = []
    # Indices = array of labels and each label is an array of indices
    indices = create_indices(imgs)

    #for x in tqdm(range(num_triplets)):
    for x in range(num_triplets):
        c1 = np.random.randint(0, n_classes)
        c2 = np.random.randint(0, n_classes)
        while len(indices[c1]) < 2:
            c1 = np.random.randint(0, n_classes)

        while c1 == c2:
            c2 = np.random.randint(0, n_classes)
        if len(indices[c1]) == 2:  # hack to speed up process
            n1, n2 = 0, 1
        else:
            n1 = np.random.randint(0, len(indices[c1]) - 1)
            n2 = np.random.randint(0, len(indices[c1]) - 1)
            while n1 == n2:
                n2 = np.random.randint(0, len(indices[c1]) - 1)
        if len(indices[c2]) ==1:
            n3 = 0
        else:
            n3 = np.random.randint(0, len(indices[c2]) - 1)

        triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3],c1,c2])
    return triplets



class DeepSpeakerDataset(data.Dataset):

    def __init__(self, voxceleb, dir, n_triplets,loader, transform=None, *arg, **kw):

        print('Looking for audio [wav] files in {}.'.format(dir))
        #voxceleb = read_voxceleb_structure(dir)

        #voxceleb = voxceleb[voxceleb['subset'] == 'dev']

        #voxceleb = voxceleb[1:5000]
        #voxceleb = voxceleb[445:448]

        if len(voxceleb) == 0:
            raise(RuntimeError(('Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh')))

        classes, class_to_idx = find_classes(voxceleb)
        imgs = []
        for vox_item in voxceleb.iterrows():
            item = (dir +'/voxceleb1_wav/' + vox_item[1]['filename']+'.wav', class_to_idx[vox_item[1]['speaker_id']])
            imgs.append(item)

        self.root = dir
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader

        self.n_triplets = n_triplets

        print('Generating {} triplets'.format(self.n_triplets))
        self.training_triplets = generate_triplets(self.imgs, self.n_triplets,len(self.classes))



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

        # Get the index of each image in the triplet
        a, p, n,c1,c2 = self.training_triplets[index]

        # transform images if required
        img_a, img_p, img_n = transform(a), transform(p), transform(n)
        return img_a, img_p, img_n,c1,c2

    def __len__(self):
        return len(self.training_triplets)