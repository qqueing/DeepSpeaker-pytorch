#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: extract_vector.py
@Time: 19-6-25 下午3:47
@Overview:Given audio samples, extract embeddings from checkpoint file in this script.
Extractor vectors for enrollment and test sets.
For enrollment set: Output (features, spkid)
For test set: Output (features, uttid)

"""
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
import pdb

import numpy as np
from tqdm import tqdm
from Define_Model import ResSpeakerModel
from logger import Logger

from Process_Data.DeepSpeakerDataset_dynamic import DeepSpeakerEnrollDataset
from Process_Data.voxceleb_wav_reader import if_load_npy

from Define_Model import PairwiseDistance
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, truncatedinputfromMFB,read_MFB,read_audio,mk_MFB
from Process_Data.audio_processing import to4tensor, concateinputfromMFB

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition Feature Extraction')

# Dataset and model file path
parser.add_argument('--dataroot', type=str, default='Data/dataset/enroll',
                    help='path to extracting dataset')
parser.add_argument('--enroll', action='store_true', default=True,
                    help='enroll step or test step')
parser.add_argument('--extract-path', type=str, default='Data/xvector/enroll',
                    help='path to pairs file')
parser.add_argument('--log-dir', default='Data/extract_feature_logs',
                    help='folder to output model checkpoints')
parser.add_argument('--model-path', default='Data/checkpoint/checkpoint_35.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Model options
parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--n-triplets', type=int, default=100000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=0.1, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--min-softmax-epoch', type=int, default=2, metavar='MINEPOCH',
                    help='minimum epoch for initial parameter using softmax (default: 2')
parser.add_argument('--loss-ratio', type=float, default=2.0, metavar='LOSSRATIO',
                    help='the ratio softmax loss - triplet loss (default: 2.0')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='adagrad', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=1, metavar='LI',
                    help='how many batches to wait before logging training status')

# Spectrum feature options
parser.add_argument('--mfb', action='store_true', default=True,
                    help='start from MFB file')
parser.add_argument('--makemfb', action='store_true', default=False,
                    help='need to make mfb file')

args = parser.parse_args()


# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.cuda:
    cudnn.benchmark = True
CKP_DIR = args.model_path
EXT_DIR = args.extract_path
LOG_DIR = args.log_dir + '/extract_{}-n{}-lr{}-wd{}-m{}-embed{}-alpha10'\
    .format(args.optimizer, args.n_triplets, args.lr, args.wd,
            args.margin, args.embedding_size)

data_set_list = "Data/enroll_set.npy"
classes_to_label_list = "Data/enroll_classes.npy"
dataroot = args.dataroot

if not args.enroll:
    dataroot = args.dataroot.replace("enroll", "test")
    EXT_DIR = args.extract_path.replace("enroll", "test")
    data_set_list = data_set_list.replace("enroll", "test")
    classes_to_label_list = classes_to_label_list.replace("enroll", "test")

if not os.path.exists(EXT_DIR):
    os.makedirs(EXT_DIR)
# create logger
logger = Logger(LOG_DIR)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
l2_dist = PairwiseDistance(2)


audio_set = []
audio_set = if_load_npy(dataroot, data_set_list)

if args.makemfb:
    #pbar = tqdm(voxceleb)
    for datum in audio_set:
        # print(datum['filename'])
        mk_MFB((datum['filename']+'.wav'))
    print("Complete convert")

if args.mfb:
    transform = transforms.Compose([
        concateinputfromMFB(),
        to4tensor()
        # truncatedinputfromMFB(),
        # totensor()
    ])
    transform_T = transforms.Compose([
        truncatedinputfromMFB(input_per_file=args.test_input_per_file),
        totensor()
    ])
    file_loader = read_MFB
else:
    transform = transforms.Compose([
                        truncatedinput(),
                        toMFB(),
                        totensor(),
                        #tonormal()
                    ])
    file_loader = read_audio

enroll_dir = DeepSpeakerEnrollDataset(audio_set=audio_set, dir=args.dataroot, loader=file_loader, transform=transform, enroll=args.enroll)

classes_to_label = enroll_dir.class_to_idx
if not os.path.isfile(classes_to_label_list):
    if not args.enroll:
        classes_to_label = enroll_dir.uttid
    np.save(classes_to_label_list, classes_to_label)
    print("update the classes to labels list files.")
else:
    # TODO: add new classes to the file
    print("Classes to labels list files already existed!")

try:
    qwer = enroll_dir.__getitem__(3)
except IndexError:
    print("wav in enroll set is less than 3?")

del audio_set
# pdb.set_trace()

def main():
    test_display_triplet_distance = False

    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    print('\nNumber of Wav file:\n{}\n'.format(len(enroll_dir.indices)))

    # instantiate model and initialize weights
    model = ResSpeakerModel(embedding_size=args.embedding_size, resnet_size=10, num_classes=1211)

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    if args.model_path:
        if os.path.isfile(args.model_path):
            print('=> loading checkpoint {}'.format(args.model_path))
            checkpoint = torch.load(args.model_path, map_location='cpu')
            args.start_epoch = checkpoint['epoch']

            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}

            model.load_state_dict(filtered)

            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            raise Exception('=> no checkpoint found at {}'.format(args.model_path))

    # train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    epoch = args.start_epoch

    def my_collate(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        target = torch.LongTensor(target)
        return [data, target]

    enroll_loader = torch.utils.data.DataLoader(enroll_dir, batch_size=args.test_batch_size, collate_fn=my_collate, shuffle=False, **kwargs)
    #for epoch in range(start, end):

    enroll(enroll_loader, model, epoch)


def enroll(enroll_loader, model, epoch):
    # switch to evaluate mode
    # pdb.set_trace()
    model.eval()
    labels, features = [], []

    pbar = tqdm(enumerate(enroll_loader))
    for batch_idx, (data_a, label) in pbar:

        pdb.set_trace()
        current_sample = data_a.size(0)
        data_a = data_a.resize_(args.test_input_per_file * current_sample, 1, data_a.size(2), data_a.size(3))
        if args.cuda:
            data_a = data_a.cuda(),
        data_a, label = Variable(data_a, volatile=True),  Variable(label)

        # compute output
        out_a = model(data_a)

        features.append(out_a)
        if not args.enroll:
            labels.append(label)
        else:
            labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('{}: {} [{}/{} ({:.0f}%)]'.format(
                "enroll" if args.enroll else "test",
                epoch,
                batch_idx * len(data_a),
                len(enroll_loader.dataset),
                100. * batch_idx / len(enroll_loader)))

    print('Xvector extraction completed!')
    feature_np = []
    for tensors in features:
        for tensor in tensors:
            feature_np.append(tensor)

    label_np = []
    for label in labels:
       for lab in label:
            label_np.append(lab)
    wav_dict = []
    for index, label in enumerate(label_np):
        wav_dict.append((label, feature_np[index]))

    # wav_dict = dict(zip(label_np, feature_np))

    np.save(EXT_DIR+'/extract_{}-lr{}-wd{}-embed{}-alpha10.npy'.format(args.optimizer, args.lr, args.wd, args.embedding_size), wav_dict)

    logger.log_value('Extracted Num', len(features))


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer

if __name__ == '__main__':
    main()




