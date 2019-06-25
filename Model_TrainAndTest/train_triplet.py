#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os


import numpy as np
from tqdm import tqdm
from Model_Define.model import DeepSpeakerModel
from eval_metrics import evaluate
from logger import Logger

#from DeepSpeakerDataset_static import DeepSpeakerDataset
from Dataset_Process.DeepSpeakerDataset_dynamic import DeepSpeakerDataset
from Dataset_Process.VoxcelebTestset import VoxcelebTestset
from Dataset_Process.voxceleb_wav_reader import read_my_voxceleb_structure

from Model_Define.model import PairwiseDistance,TripletMarginLoss
from Dataset_Process.audio_processing import toMFB, totensor, truncatedinput, truncatedinputfromMFB,read_MFB,read_audio,mk_MFB
# Version conflict

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
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Model options
parser.add_argument('--dataroot', type=str, default='Data/dataset',
                    help='path to dataset')
parser.add_argument('--test-pairs-path', type=str, default='Data/dataset/ver_list.txt',
                    help='path to pairs file')

parser.add_argument('--log-dir', default='Data/pytorch_speaker_logs',
                    help='folder to output model checkpoints')

parser.add_argument('--ckp-dir', default='Data/checkpoint',
                    help='folder to output model checkpoints')

parser.add_argument('--resume',
                    default='Data/checkpoint/checkpoint_10k_21.pth',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=29, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--batch-size', type=int, default=512, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=8, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

#parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
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
CKP_DIR = args.ckp_dir
LOG_DIR = args.log_dir + '/run-optim_{}-n{}-lr{}-wd{}-m{}-embeddings{}-msceleb-alpha10'\
    .format(args.optimizer, args.n_triplets, args.lr, args.wd,
            args.margin,args.embedding_size)

# create logger
logger = Logger(LOG_DIR)


kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
l2_dist = PairwiseDistance(2)

# Read file list from dataset path
print("================================Reading Dataset File List==================================")
voxceleb_list = "Data/voxceleb.npy"
voxceleb_dev_list = "Data/voxceleb_dev.npy"
voxceleb_dev_10k_list = "Data/voxceleb_dev_10k.npy"

voxceleb = []
voxceleb_dev = []
voxceleb_dev_10k = []
if os.path.isfile(voxceleb_list):
    voxceleb = np.load(voxceleb_list, allow_pickle=True)
else:
    voxceleb = read_my_voxceleb_structure(args.dataroot)
    np.save(voxceleb_list, voxceleb)

# Make fbank feature if not yet.
if args.makemfb:
    #pbar = tqdm(voxceleb)
    for datum in voxceleb:
        # print(datum['filename'])
        mk_MFB((args.dataroot +'/voxceleb1_wav/' + datum['filename']+'.wav'))
    print("Complete convert")

# Create file loader for dataset
if args.mfb:
    transform = transforms.Compose([
        truncatedinputfromMFB(),
        totensor()
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
print("Creating file loader for dataset completed!")

# Get the file list of development set
if os.path.isfile(voxceleb_dev_list):
    voxceleb_dev = np.load(voxceleb_dev_list, allow_pickle=True)
    voxceleb_dev_10k = voxceleb_dev[:10000]
else:
    voxceleb_dev = [datum for datum in voxceleb if datum['subset']=='dev']
    np.save(voxceleb_dev_list, voxceleb_dev)

    voxceleb_dev_10k = voxceleb_dev[:10000]
    np.save(voxceleb_dev_10k_list, voxceleb_dev_10k)


# Reduce the dev set
voxceleb_dev = voxceleb_dev_10k

train_dir = DeepSpeakerDataset(voxceleb = voxceleb_dev, dir=args.dataroot,n_triplets=args.n_triplets,loader = file_loader,transform=transform)

# Remove the reference to reduce memory usage
del voxceleb
del voxceleb_dev

test_dir = VoxcelebTestset(dir=args.dataroot, pairs_path=args.test_pairs_path, loader = file_loader, transform=transform_T)
# Test if there is data in test dir
qwer = test_dir.__getitem__(3)


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    test_display_triplet_distance = False

    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    print('\nNumber of Classes:\n{}\n'.format(len(train_dir.classes)))

    # instantiate model and initialize weights
    model = DeepSpeakerModel(embedding_size=args.embedding_size,
                      num_classes=len(train_dir.classes))

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)

            # Filter that remove uncessary component in checkpoint file
            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}

            model.load_state_dict(filtered)

            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    print('start epoch is : ' + str(start))
    #start = 0
    end = start + args.epochs

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    for epoch in range(start, end):

        train(train_loader, model, optimizer, epoch)
        test(test_loader, model, epoch)
        #break;


def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()

    labels, distances = [], []

    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data_a, data_p, data_n,label_p,label_n) in pbar:
        #print("on training{}".format(epoch))
        data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
        data_a, data_p, data_n = Variable(data_a), Variable(data_p), \
                                 Variable(data_n)

        # compute output
        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)


        if epoch > args.min_softmax_epoch:
            triplet_loss = TripletMarginLoss(args.margin).forward(out_a, out_p, out_n)
            loss = triplet_loss
            # compute gradient and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.log_value('selected_triplet_loss', triplet_loss.data[0]).step()
            #logger.log_value('selected_cross_entropy_loss', cross_entropy_loss.data[0]).step()
            logger.log_value('selected_total_loss', loss.data[0]).step()

            if batch_idx % args.log_interval == 0:
                pbar.set_description(
                    'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_a), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.data[0]))


            dists = l2_dist.forward(out_a,out_n) #torch.sqrt(torch.sum((out_a - out_n) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy())
            labels.append(np.zeros(dists.size(0)))


            dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy())
            labels.append(np.ones(dists.size(0)))


        else:
        # Choose the hard negatives
            d_p = l2_dist.forward(out_a, out_p)
            d_n = l2_dist.forward(out_a, out_n)
            all = (d_n - d_p < args.margin).cpu().data.numpy().flatten()

            # log loss value for mini batch.
            total_coorect = np.where(all == 0)
            logger.log_value('Minibatch Train Accuracy', len(total_coorect[0]))

            total_dist = (d_n - d_p).cpu().data.numpy().flatten()
            logger.log_value('Minibatch Train distance', np.mean(total_dist))

            hard_triplets = np.where(all == 1)
            if len(hard_triplets[0]) == 0:
                continue
            out_selected_a = Variable(torch.from_numpy(out_a.cpu().data.numpy()[hard_triplets]).cuda())
            out_selected_p = Variable(torch.from_numpy(out_p.cpu().data.numpy()[hard_triplets]).cuda())
            out_selected_n = Variable(torch.from_numpy(out_n.cpu().data.numpy()[hard_triplets]).cuda())

            selected_data_a = Variable(torch.from_numpy(data_a.cpu().data.numpy()[hard_triplets]).cuda())
            selected_data_p = Variable(torch.from_numpy(data_p.cpu().data.numpy()[hard_triplets]).cuda())
            selected_data_n = Variable(torch.from_numpy(data_n.cpu().data.numpy()[hard_triplets]).cuda())

            selected_label_p = torch.from_numpy(label_p.cpu().numpy()[hard_triplets])
            selected_label_n= torch.from_numpy(label_n.cpu().numpy()[hard_triplets])
            triplet_loss = TripletMarginLoss(args.margin).forward(out_selected_a, out_selected_p, out_selected_n)

            cls_a = model.forward_classifier(selected_data_a)
            cls_p = model.forward_classifier(selected_data_p)
            cls_n = model.forward_classifier(selected_data_n)

            criterion = nn.CrossEntropyLoss()
            predicted_labels = torch.cat([cls_a,cls_p,cls_n])
            true_labels = torch.cat([Variable(selected_label_p.cuda()),Variable(selected_label_p.cuda()),Variable(selected_label_n.cuda())])

            cross_entropy_loss = criterion(predicted_labels.cuda(),true_labels.cuda())

            loss = cross_entropy_loss + triplet_loss * args.loss_ratio
            # compute gradient and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # log loss value for hard selected sample
            logger.log_value('selected_triplet_loss', triplet_loss.data[0]).step()
            logger.log_value('selected_cross_entropy_loss', cross_entropy_loss.data[0]).step()
            logger.log_value('selected_total_loss', loss.data[0]).step()
            if batch_idx % args.log_interval == 0:
                pbar.set_description(
                    'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f} \t # of Selected Triplets: {:4d}'.format(
                        epoch, batch_idx * len(data_a), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.data[0],len(hard_triplets[0])))


            dists = l2_dist.forward(out_selected_a,out_selected_n) #torch.sqrt(torch.sum((out_a - out_n) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy())
            labels.append(np.zeros(dists.size(0)))


            dists = l2_dist.forward(out_selected_a,out_selected_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy())
            labels.append(np.ones(dists.size(0)))


    #accuracy for hard selected sample, not all sample.
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val, far = evaluate(distances,labels)
    print('\33[91mTrain set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
    logger.log_value('Train Accuracy', np.mean(accuracy))

    # do checkpointing
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               '{}/checkpoint_10k_{}.pth'.format(CKP_DIR, epoch))


def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        current_sample = data_a.size(0)
        data_a = data_a.resize_(args.test_input_per_file *current_sample, 1, data_a.size(2), data_a.size(3))
        data_p = data_p.resize_(args.test_input_per_file *current_sample, 1, data_a.size(2), data_a.size(3))
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        # compute output
        out_a, out_p = model(data_a), model(data_p)
        dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        dists = dists.data.cpu().numpy()
        dists = dists.reshape(current_sample,args.test_input_per_file).mean(axis=1)
        distances.append(dists)
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    #print("distance {.8f}".format(distances))
    #print("distance {.1f}".format(labels))
    tpr, fpr, accuracy, val,  far = evaluate(distances,labels)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
    logger.log_value('Test Accuracy', np.mean(accuracy))


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
