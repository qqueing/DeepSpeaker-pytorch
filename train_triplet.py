from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn
import os


import numpy as np
from tqdm import tqdm
from model import DeepSpeakerModel
from eval_metrics import evaluate
from logger import Logger

from DeepSpeakerDataset import DeepSpeakerDataset
from VoxcelebTestset import VoxcelebTestset
from voxceleb_wav_reader import read_voxceleb_structure

from model import PairwiseDistance
from audio_processing import toMFB, totensor, truncatedinput, truncatedinputfromMFB,read_MFB,read_audio,mk_MFB

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Model options
parser.add_argument('--dataroot', type=str, default='./voxceleb',
                    help='path to dataset')
parser.add_argument('--test-pairs-path', type=str, default='./voxceleb/voxceleb1_test3.txt',
                    help='path to pairs file')

parser.add_argument('--log-dir', default='./data/pytorch_speaker_logs',
                    help='folder to output model checkpoints')

parser.add_argument('--resume',
                    default=None,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--batch-size', type=int, default=512, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='BST',
                    help='input batch size for testing (default: 1000)')
#parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
                    help='how many triplets will generate from the dataset')

parser.add_argument('--margin', type=float, default=0.1, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')

parser.add_argument('--min-softmax-epoch', type=int, default=5, metavar='MINEPOCH',
                    help='minimum epoch for initial parameter using softmax (default: 5')

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

LOG_DIR = args.log_dir + '/run-optim_{}-n{}-lr{}-wd{}-m{}-embeddings{}-msceleb-alpha10'\
    .format(args.optimizer, args.n_triplets, args.lr, args.wd,
            args.margin,args.embedding_size)

# create logger
logger = Logger(LOG_DIR)


class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
l2_dist = PairwiseDistance(2)


voxceleb = read_voxceleb_structure(args.dataroot)
if args.makemfb:
    for datum in voxceleb.iterrows():
        mk_MFB((args.dataroot +'/voxceleb1_wav/' + datum[1]['filename']+'.wav'))

if args.mfb:
    transform = transforms.Compose([
        truncatedinputfromMFB(),
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




voxceleb_dev = voxceleb.loc[lambda voxceleb: voxceleb.subset == 'dev']


train_dir = DeepSpeakerDataset(voxceleb = voxceleb_dev, dir=args.dataroot,n_triplets=args.n_triplets,loader = file_loader,transform=transform)

test_dir = VoxcelebTestset(dir=args.dataroot,pairs_path=args.test_pairs_path,loader = file_loader, transform=transform)

#qwer = train_dir.__getitem__(3)


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
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    end = start + args.epochs

    for epoch in range(start, end):
        train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
        train(train_loader, model, optimizer, epoch)

        test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
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
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))


def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        # compute output
        out_a, out_p = model(data_a), model(data_p)
        dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
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