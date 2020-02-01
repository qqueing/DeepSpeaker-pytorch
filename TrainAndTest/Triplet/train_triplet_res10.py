"""
@Overview:
Implement Training ResNet 10 for Speaker Verification!
    Enrollment set files will be in the 'Data/enroll_set.npy' and the classes-to-index file is 'Data/enroll_classes.npy'
    Test set files are in the 'Data/test_set.npy' and the utterances-to-index file is 'Data/test_classes.npy'.

    Training the net with Softmax for 10 epoch, and 15 epoch with Triplet Loss. The distance funtion for triplet loss is cosine similarity.
    Add train accuracy on the train set for softmax training.
"""
#from __future__ import print_function
import argparse
import pathlib
import pdb
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os

import numpy as np
from tqdm import tqdm
from Define_Model.model import DeepSpeakerModel
from eval_metrics import evaluate_eer, evaluate_kaldi_eer
from logger import Logger

#from DeepSpeakerDataset_static import DeepSpeakerDataset
from Process_Data.DeepSpeakerDataset_dynamic import DeepSpeakerDataset, ClassificationDataset
from Process_Data.VoxcelebTestset import VoxcelebTestset
from Process_Data.voxceleb_wav_reader import wav_list_reader

from Process_Data.audio_processing import toMFB, totensor, truncatedinput, truncatedinputfromMFB, read_MFB, read_audio, \
    mk_MFB, concateinputfromMFB, varLengthFeat, PadCollate, TripletPadCollate

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

import warnings
warnings.filterwarnings("ignore")


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Model options
parser.add_argument('--dataroot', type=str, default='/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/fbank64',
                    help='path to dataset')
parser.add_argument('--test-dataroot', type=str, default='/home/cca01/work2019/yangwenhao/mydataset/voxceleb1/fbank64',
                    help='path to voxceleb1 test dataset')
parser.add_argument('--test-pairs-path', type=str, default='Data/dataset/ver_list.txt',
                    help='path to pairs file')

parser.add_argument('--log-dir', default='Data/pytorch_speaker_logs',
                    help='folder to output model checkpoints')
parser.add_argument('--ckp-dir', default='Data/checkpoint/Triplet/ResCNN10',
                    help='folder to output model checkpoints')
parser.add_argument('--resume',
                    default='Data/checkpoint/Triplet/ResCNN10/checkpoint_1.pth',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=15, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--cos-sim', action='store_true', default=False,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--resnet-size', type=int, default=10, metavar='BS',
                    help='resnet size for training (default: 34)')
parser.add_argument('--batch-size', type=int, default=100, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

#parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
parser.add_argument('--n-triplets', type=int, default=960000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=0.1, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--min-softmax-epoch', type=int, default=3, metavar='MINEPOCH',
                    help='minimum epoch for initial parameter using softmax (default: 2')
parser.add_argument('--loss-ratio', type=float, default=2.0, metavar='LOSSRATIO',
                    help='the ratio softmax loss - triplet loss (default: 2.0')

parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
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
parser.add_argument('--gpu-id', default='2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--data-parallel', action='store_true', default=True,
                    help='using Cosine similarity')

parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
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
torch.manual_seed(args.seed)

if args.cuda:
    cudnn.benchmark = True

# create logger
writer = SummaryWriter(logdir=args.ckp_dir, filename_suffix='ada_0.05')
kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

l2_dist = nn.PairwiseDistance()
# l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6)

# Read file list from dataset path
print("==========================================Reading Dataset File List==================================")
voxceleb, train_set, valid_set = wav_list_reader(args.dataroot, split=True)
# voxceleb2, voxceleb_dev2 = voxceleb2_list_reader(args.dataroot)

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
        varLengthFeat(),
        # concateinputfromMFB(),
        # truncatedinputfromMFB(),
        totensor()
    ])
    transform_T = transforms.Compose([
        concateinputfromMFB(input_per_file=args.test_input_per_file),
        # truncatedinputfromMFB(input_per_file=args.test_input_per_file),
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
print("==============================Creating file loader for dataset completed!=============================")

# Reduce the dev set
#voxceleb_dev = voxceleb_dev_10k

train_dir = DeepSpeakerDataset(voxceleb=train_set, dir=args.dataroot, n_triplets=args.n_triplets, loader = file_loader, transform=transform)
test_dir = VoxcelebTestset(dir=args.dataroot, pairs_path=args.test_pairs_path, loader=file_loader, transform=transform_T)
valid_dir = ClassificationDataset(voxceleb=valid_set, dir=args.dataroot, loader=file_loader, transform=transform)

# Remove the reference to reduce memory usage
del voxceleb
del train_set
del valid_set

# Test if there is data in test dir
# qwer = test_dir.__getitem__(3)

def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive

    # print the experiment configuration
    print('\33[91m\nCurrent time is {}.\33[0m'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))
    print('Number of Speakers: {}\n'.format(len(train_dir.classes)))

    # instantiate model and initialize weights
    model = DeepSpeakerModel(resnet_size=10,
                             embedding_size=args.embedding_size,
                            num_classes=len(train_dir.classes))

    if args.cuda:
        model.cuda()

    if args.data_parallel:
        model = torch.nn.DataParallel(model, device_ids=[2, 3])

    optimizer = create_optimizer(model, args.lr)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
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

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=True, collate_fn=TripletPadCollate(dim=2), **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=args.test_batch_size, shuffle=False, collate_fn=PadCollate(dim=2), **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    for epoch in range(start, end):
        train(train_loader, model, optimizer, epoch)
        test(test_loader, valid_loader, model, epoch)
        # break

    writer.close()


def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()

    # labels, distances = [], []
    correct = 0.
    total_datasize = 0.
    total_loss = 0.
    pbar = tqdm(enumerate(train_loader))

    triplet = nn.TripletMarginLoss(margin=args.margin, p=2.0, eps=1e-06)
    cross_entropy = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    for batch_idx, (data_a, data_p, data_n, label_p, label_n) in pbar:
        #print("on training{}".format(epoch))
        data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
        data_a, data_p, data_n = Variable(data_a), Variable(data_p), \
                                 Variable(data_n)

        # compute output
        out_a = model(data_a)
        out_p = model(data_p)
        out_n = model(data_n)

        if epoch > args.min_softmax_epoch:
            # Choose the hard negatives
            # Todo: choose semi-hard triplet pairs
            # d_p = l2_dist(out_a, out_p)
            # d_n = l2_dist(out_a, out_n)
            #
            # # hard pairs
            # # all = (d_p - d_n < args.margin).cpu().data.numpy().flatten()
            #
            # # semi-hard pairs
            # all = (d_p - d_n < 0).cpu().data.numpy().flatten()
            # hard_triplets = np.where(all == 1)
            #
            # if len(hard_triplets[0]) == 0:
            #     continue
            #
            # out_selected_a = Variable(torch.from_numpy(out_a.cpu().data.numpy()[hard_triplets]).cuda())
            # out_selected_p = Variable(torch.from_numpy(out_p.cpu().data.numpy()[hard_triplets]).cuda())
            # out_selected_n = Variable(torch.from_numpy(out_n.cpu().data.numpy()[hard_triplets]).cuda())
            #
            # loss = triplet_loss(out_selected_a, out_selected_p, out_selected_n)

            loss = triplet(out_a, out_p, out_n)
            total_loss += loss.item()
            # pdb.set_trace()
            # compute gradient and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cls_a = model.forward_classifier(out_a)
            cls_p = model.forward_classifier(out_p)
            cls_n = model.forward_classifier(out_n)

            predicted_labels = torch.cat([cls_a, cls_p, cls_n])
            true_labels = torch.cat([Variable(label_p.cuda()), Variable(label_p.cuda()), Variable(label_n.cuda())])
            predicted_one_labels = softmax(predicted_labels)
            predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]

            # pdb.set_trace()
            total_loss += loss.item()
            batch_correct = float((predicted_one_labels.cuda() == true_labels.cuda()).sum().item())
            minibatch_acc = float(batch_correct) / len(predicted_one_labels)

            correct += batch_correct
            total_datasize += len(predicted_one_labels)

            if batch_idx % args.log_interval == 0:
                pbar.set_description(
                    'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tBatch Loss: {:.6f}\tMinibatch Accuracy: {:.4f}%\tSelected Triplets: {:4d}'.format(
                        epoch,
                        batch_idx * len(data_a),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item(),
                        100. * minibatch_acc,
                        len(data_a)))

        else:
            d_p = l2_dist.forward(out_a, out_p)
            d_n = l2_dist.forward(out_a, out_n)
            all = (d_n - d_p < args.margin).cpu().data.numpy().flatten()

            hard_triplets = np.where(all == 1)
            if len(hard_triplets[0]) == 0:
                continue

            out_selected_a = Variable(torch.from_numpy(out_a.cpu().data.numpy()[hard_triplets]).cuda())
            out_selected_p = Variable(torch.from_numpy(out_p.cpu().data.numpy()[hard_triplets]).cuda())
            out_selected_n = Variable(torch.from_numpy(out_n.cpu().data.numpy()[hard_triplets]).cuda())

            selected_label_p = torch.from_numpy(label_p.cpu().numpy()[hard_triplets])
            selected_label_n = torch.from_numpy(label_n.cpu().numpy()[hard_triplets])

            triplet_loss = triplet(out_selected_a, out_selected_p, out_selected_n)

            cls_a = model.forward_classifier(out_selected_a)
            cls_p = model.forward_classifier(out_selected_p)
            cls_n = model.forward_classifier(out_selected_n)

            predicted_labels = torch.cat([cls_a, cls_p, cls_n])
            true_labels = torch.cat([Variable(selected_label_p.cuda()), Variable(selected_label_p.cuda()),
                                     Variable(selected_label_n.cuda())])

            cross_entropy_loss = cross_entropy(predicted_labels.cuda(), true_labels.cuda())

            loss = cross_entropy_loss + triplet_loss * args.loss_ratio
            # compute gradient and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #pdb.set_trace()
            predicted_one_labels = softmax(predicted_labels)
            predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]

            batch_correct = (predicted_one_labels.cuda() == true_labels.cuda()).sum().item()
            minibatch_acc = float(batch_correct/len(predicted_one_labels))
            correct += batch_correct
            total_datasize += len(predicted_one_labels)
            total_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tBatch Loss: {:.4f}\tBatch Accuracy: {:.4f}%'.format(
                        epoch,
                        batch_idx * len(data_a),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item(),
                        100. * minibatch_acc
                    ))

    check_path = pathlib.Path('{}/checkpoint_{}.pth'.format(args.ckp_dir, epoch))
    if not check_path.parent.exists():
        os.makedirs(str(check_path.parent))

    # do checkpointing
    torch.save({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                 str(check_path))

    predict_accuracy = correct / total_datasize

    # writer.add_scalar('Train/Train_EER', eer, epoch)
    writer.add_scalar('Train/Accuracy', predict_accuracy, epoch)
    writer.add_scalar('Train/Loss', total_loss/len(train_loader), epoch)

    print('\33[91mTrain set Accuracy: {:.8f} Average loss is {}\33[0m\n'.format(predict_accuracy, total_loss/len(train_loader)))


def test(test_loader, valid_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    valid_pbar = tqdm(enumerate(valid_loader))
    softmax = nn.Softmax(dim=1)

    correct = 0.
    total_datasize = 0.

    for batch_idx, (data, label) in valid_pbar:
        model.eval()

        data = Variable(data.cuda())
        # compute output
        out = model(data)
        cls = model.forward_classifier(out)

        predicted_labels = cls
        true_labels = Variable(label.cuda())

        # pdb.set_trace()
        predicted_one_labels = softmax(predicted_labels)
        predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]

        batch_correct = (predicted_one_labels.cuda() == true_labels.cuda()).sum().item()
        minibatch_acc = float(batch_correct / len(predicted_one_labels))
        correct += batch_correct
        total_datasize += len(predicted_one_labels)

        if batch_idx % args.log_interval == 0:
            valid_pbar.set_description(
                'Test Epoch for Classification: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tBatch Accuracy: {:.4f}%'.format(
                    epoch,
                    batch_idx * len(data),
                    len(valid_loader.dataset),
                    100. * batch_idx / len(valid_loader),
                    100. * minibatch_acc
                ))

    valid_accuracy = 100. * correct/total_datasize
    writer.add_scalar('Test/Valid_Accuracy', valid_accuracy, epoch)

    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        current_sample = data_a.size(0)
        data_a = data_a.resize_(args.test_input_per_file *current_sample, 1, data_a.size(2), data_a.size(3))
        data_p = data_p.resize_(args.test_input_per_file *current_sample, 1, data_a.size(2), data_a.size(3))

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a), Variable(data_p), Variable(label)

        # compute output
        out_a, out_p = model(data_a), model(data_p)
        dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        dists = dists.data.cpu().numpy()
        dists = dists.reshape(current_sample, args.test_input_per_file).mean(axis=1)
        distances.append(dists)
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset), 100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])


    eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    writer.add_scalar('Test/EER', eer, epoch)
    writer.add_scalar('Test/Threshold', eer_threshold, epoch)


    if args.cos_sim:
        print(
            '\33[91mFor cos_distance, Test set verification ERR is {:.8f}%, when threshold is {}. Valid set classificaton accuracy is {:.2f}%.\n\33[0m'.format(
                100. * eer, eer_threshold, valid_accuracy))
    else:
        print('\33[91mFor l2_distance, Test set verification ERR is {:.8f}%, when threshold is {}. Valid set classificaton accuracy is {:.2f}%.\n\33[0m'.format(
                100. * eer, eer_threshold, valid_accuracy))


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
