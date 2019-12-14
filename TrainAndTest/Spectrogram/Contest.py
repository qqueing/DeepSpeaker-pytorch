#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Contest.py
@Time: 2019/12/13 下午4:07
@Overview:
"""
import csv
import math
import os
import random

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from python_speech_features import mfcc, logfbank, delta
from scipy.signal.windows import hamming
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

SAMPLE_RATE = 16000

VAD = False

# FEATURE = 'mfcc'
# FEATURE_LEN = 13 * 3
# WIN_LEN = 0.025
# WIN_STEP = 0.01

# FEATURE = 'logfbank'
# FEATURE_LEN = 26 * 3
# WIN_LEN = 0.025
# WIN_STEP = 0.01

FEATURE = 'fft'#特征类型
FEATURE_LEN = 161#特征维度
WIN_LEN = 0.02#滑窗窗口长度，单位s
WIN_STEP = 0.01#滑窗滑动距离，单位s

N_FFT = int(WIN_LEN * SAMPLE_RATE)#滑窗采样点
HOP_LEN = int(WIN_STEP * SAMPLE_RATE)#滑窗滑动距离采样点

N_FRAMES = 300#训练帧数
DURATION = (N_FRAMES - 1) * WIN_STEP#训练单句时长
N_SAMPLES = int(DURATION * SAMPLE_RATE)#训练单句采样点

N_TEST_FRAMES = 300#未用
TEST_DURATION = (N_TEST_FRAMES - 1) * WIN_STEP#未用
N_TEST_SAMPLES = int(TEST_DURATION * SAMPLE_RATE)#未用

TEST_WAV = '/mnt/datasets/tongdun/test_set/'
TRAIN_MANIFEST = '/home/kesci/large-work/tongdun_all_manifest.csv'

if VAD:
    TEST_FEATURE = '/home/kesci/work/test/vad/{}/'.format(FEATURE)
else:
    TEST_FEATURE = '/home/kesci/work/test/{}/'.format(FEATURE)


def make_feature(y, sr):#提取特征，y是语音data部分，sr为采样率
    if FEATURE == 'fft':#提取fft特征
        S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LEN, window=hamming)#进行短时傅里叶变换，参数意义在一开始有定义
        feature, _ = librosa.magphase(S)
        feature = np.log1p(feature)#log1p操作
        feature = feature.transpose()
    else:
        if FEATURE == 'logfbank':#提取fbank特征
            feature = logfbank(y, sr, winlen=WIN_LEN, winstep=WIN_STEP)
        else:
            feature = mfcc(y, sr, winlen=WIN_LEN, winstep=WIN_STEP)#提取mfcc特征
        feature_d1 = delta(feature, N=1)#加上两个delta，特征维度X3
        feature_d2 = delta(feature, N=2)
        feature = np.hstack([feature, feature_d1, feature_d2])#横向拼起来
    return normalize(feature)#返回归一化的特征


def normalize(v):#进行归一化，v是语音特征
    return (v - v.mean(axis=0)) / (v.std(axis=0) + 2e-12)


def process_test_dataset():#处理测试集
    print('processing test dataset...', end='')
    for filename in os.listdir(TEST_WAV):
        if filename[0] != '.':
            feature_path = os.path.join(TEST_FEATURE, filename.replace('.wav', '.npy'))
            if not os.path.exists(feature_path):
                y, sr = load_audio(os.path.join(TEST_WAV, filename))
                feature = make_feature(y)#提取特征
                np.save(feature_path, feature)#存为npy文件
    print('done')


if not VAD:#无vad的情况下，处理test文件
    os.makedirs(TEST_FEATURE, exist_ok=True)
    process_test_dataset()

class SpeakerTrainDataset(Dataset):#定义pytorch的训练数据及类
    def __init__(self, dataset, loader, transform, samples_per_speaker=1):#每个epoch每个人的语音采样数
        self.dataset = []
        current_sid = -1
        self.dataset = dataset
        self.n_classes = len(self.dataset)
        self.samples_per_speaker = samples_per_speaker
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return self.samples_per_speaker * self.n_classes#返回一个epoch的采样数

    def __getitem__(self, sid):#定义采样方式，sid为说话人id
        sid %= self.n_classes
        speaker = self.dataset[sid]  # sid 对应的所有utt
        y = []
        n_samples = 0
        while n_samples < N_SAMPLES:#当采样长度不够时，继续读取
            aid = random.randrange(0, len(speaker))
            audio = self.loader(speaker[aid]) # aid句utt
            # t, sr = audio[1], audio[2]
            if len(audio) < 100:#长度小于1，不使用
                continue
            feature = self.transform(audio)

        return feature, sid#返回特征和说话人id


class TruncatedInput(object):#以固定长度截断语音，并进行堆叠后输出
    def __init__(self, input_per_file=1):
        super(TruncatedInput, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):
        network_inputs = []
        n_frames = len(frames_features)
        for i in range(self.input_per_file):
            if n_frames < N_TEST_FRAMES:
                frames_slice = []
                left = N_TEST_FRAMES
                while left > n_frames:
                    frames_slice.append(frames_features)
                    left -= n_frames
                frames_slice.append(frames_features[:left])
                frames_slice = np.concatenate(frames_slice)
            else:
                start = random.randint(0, n_frames - N_TEST_FRAMES)
                frames_slice = frames_features[start:start + N_TEST_FRAMES]
            network_inputs.append(frames_slice)
        return np.array(network_inputs)


class ToTensor(object):#转换第二维度和第三维度坐标
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            return torch.FloatTensor(pic.transpose((0, 2, 1)))


class SpeakerTestDataset(Dataset):#定义pytorch测试数据的类
    def __init__(self, transform=None):
        self.transform = transform#是否使用某种变换，如上面的截断
        self.features = []
        self.pairID = []
        with open('/mnt/datasets/tongdun/pairs_id.txt') as f:
            pairs = f.readlines()
            for pair in pairs:
                pair = pair.strip()
                pair_list = pair.split('_')
                self.pairID.append(pair.strip())
                self.features.append((os.path.join(TEST_FEATURE, '{}.npy'.format(pair_list[0])),
                                      os.path.join(TEST_FEATURE, '{}.npy'.format(pair_list[1]))))

    def __getitem__(self, index):#获取样本的方式
        if self.transform is not None:
            return self.pairID[index], self.transform(np.load(self.features[index][0])),\
                   self.transform(np.load(self.features[index][1]))
        else:
            return self.pairID[index], np.array([np.load(self.features[index][0]).transpose()]),\
                   np.array([np.load(self.features[index][1]).transpose()])

    def __len__(self):#测试集数量
        return len(self.features)


class AngleLinear(nn.Module):#定义最后一层
    def __init__(self, in_features, out_features, m=3, phiflag=True):#输入特征维度，输出特征维度，margin超参数
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))#本层权重
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)#初始化权重，在第一维度上做normalize
        self.m = m
        self.phiflag = phiflag
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]#匿名函数,用于得到cos_m_theta

    @staticmethod
    def myphi(x, m):
        x = x * m
        return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) +\
               x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)

    def forward(self, x):#前向过程，输入x
        w = self.weight

        ww = w.renorm(2, 1, 1e-5).mul(1e5)#方向0上做normalize
        x_len = x.pow(2).sum(1).pow(0.5)
        w_len = ww.pow(2).sum(0).pow(0.5)

        cos_theta = x.mm(ww)
        cos_theta = cos_theta / x_len.view(-1, 1) / w_len.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)#由m和/cos(/theta)得到cos_m_theta
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k#得到/phi(/theta)
        else:
            theta = cos_theta.acos()#acos得到/theta
            phi_theta = self.myphi(theta, self.m)#得到/phi(/theta)
            phi_theta = phi_theta.clamp(-1*self.m, 1)#控制在-m和1之间

        cos_theta = cos_theta * x_len.view(-1, 1)
        phi_theta = phi_theta * x_len.view(-1, 1)
        output = [cos_theta, phi_theta]#返回/cos(/theta)和/phi(/theta)
        return output


class AngleLoss(nn.Module):#设置loss，超参数gamma，最小比例，和最大比例
    def __init__(self, gamma=0, lambda_min=5, lambda_max=1500):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def forward(self, x, y): #分别是output和target
        self.it += 1
        cos_theta, phi_theta = x #output包括上面的[cos_theta, phi_theta]
        y = y.view(-1, 1)

        index = cos_theta.data * 0.0
        index.scatter_(1, y.data.view(-1, 1), 1)#将label存成稀疏矩阵
        index = index.byte()
        index = Variable(index)

        lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.it))#动态调整lambda，来调整cos(\theta)和\phi(\theta)的比例
        output = cos_theta * 1.0
        output[index] -= cos_theta[index]*(1.0+0)/(1 + lamb)#减去目标\cos(\theta)的部分
        output[index] += phi_theta[index]*(1.0+0)/(1 + lamb)#加上目标\phi(\theta)的部分

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, y)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


class ReLU20(nn.Hardtanh):#relu
    def __init__(self, inplace=False):
        super(ReLU20, self).__init__(0, 20, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


def conv3x3(in_planes, out_planes, stride=1):#3x3卷积，输入通道，输出通道，stride
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):#定义block

    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):#输入通道，输出通道，stride，下采样
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = ReLU20(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out#block输出


class ResNet(nn.Module):#定义resnet
    def __init__(self, layers, block=BasicBlock, embedding_size=None, n_classes=1000, m=3):#block类型，embedding大小，分类数，maigin大小
        super(ResNet, self).__init__()
        if embedding_size is None:
            embedding_size = n_classes

        self.relu = ReLU20(inplace=True)

        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.in_planes = 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])

        self.in_planes = 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])

        # self.in_planes = 512
        # self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False)
        # self.bn4 = nn.BatchNorm2d(512)
        # self.layer4 = self._make_layer(block, 512, layers[3])

        self.avg_pool = nn.AdaptiveAvgPool2d([4, 1])

        self.fc = nn.Sequential(
            nn.Linear(self.in_planes * 4, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

        self.angle_linear = AngleLinear(in_features=embedding_size, out_features=n_classes, m=m)

        for m in self.modules():#对于各层参数的初始化
            if isinstance(m, nn.Conv2d):#以2/n的开方为标准差，做均值为0的正态分布
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):#weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):#weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)

        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        # x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        logit = self.angle_linear(x)
        return logit, x#返回最后一层和倒数第二层的表示


class Args:
    def __init__(self):
        self.embedding_size = 1024
        self.m = 3
        self.lambda_min = 5 #lambda下限
        self.lambda_max = 1000 #lambda 上限

        self.samples_per_speaker = 8 #每个说话人采样数
        self.epochs = 20
        self.batch_size = 128
        self.optimizer = 'sgd'
        self.momentum = 0.9
        self.dampening = 0
        self.lr = 1e-1
        self.lr_decay = 0
        self.wd = 0
        self.model_dir = '/home/kesci/work/model/asoftmax/embedding/sgd/'
        self.final_dir = '/home/kesci/large-work/model/asoftmax/embedding/sgd/'
        self.start = None
        self.resume = self.final_dir + 'net.pth'
        self.load_it = True
        self.it = None
        self.load_optimizer = True
        self.reset_linear = False
        self.seed = 123456

        self.use_out = True
        self.use_embedding = True

        self.test_batch_size = 24
        self.transform = transforms.Compose([
            TruncatedInput(input_per_file=1),
            ToTensor(),
        ])


args = Args()
device = torch.device('cuda')
os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.final_dir, exist_ok=True)


def adjust_learning_rate(optimizer, epoch):#调整学习率策略，优化器，目前轮数
    if epoch <= 15:
        lr = args.lr
    elif epoch <= 20:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch, model, criterion, optimizer, train_loader):#训练轮数，模型，loss，优化器，数据集读取
    model.train()#初始化模型为训练模式
    adjust_learning_rate(optimizer, epoch)#调整学习率

    sum_loss, sum_samples = 0, 0
    progress_bar = tqdm(enumerate(train_loader))
    for batch_idx, (data, label) in progress_bar:
        sum_samples += len(data)
        data, label = Variable(data).to(device), Variable(label).to(device)#数据和标签

        out, _ = model(data, label)#通过模型，输出最后一层和倒数第二层

        loss = criterion(out, label)#loss
        optimizer.zero_grad()
        loss.backward()#bp训练
        optimizer.step()

        sum_loss += loss.item() * len(data)
        progress_bar.set_description(
            'Train Epoch: {:3d} [{:4d}/{:4d} ({:3.3f}%)] Loss: {:.4f}'.format(
                epoch, batch_idx + 1, len(train_loader),
                100. * (batch_idx + 1) / len(train_loader),
                sum_loss / sum_samples))

    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'it': criterion.it,
                'optimizer': optimizer.state_dict()},
               '{}/net_{}.pth'.format(args.model_dir, epoch))#保存当轮的模型到net_{}.pth
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'it': criterion.it,
                'optimizer': optimizer.state_dict()},
               '{}/net.pth'.format(args.final_dir))#保存当轮的模型到net.pth


def test(model, test_loader):#测试，模型，测试集读取
    model.eval()#设置为测试模式

    pairs, similarities_out, similarities_embedding = [], [], []
    progress_bar = tqdm(enumerate(test_loader))
    for batch_idx, (pair, data1, data2) in progress_bar:#按batch读取数据
        pairs.append(pair)
        with torch.no_grad():
            data1, data2 = Variable(data1).to(device), Variable(data2).to(device)

            out1, embedding1 = model(data1)
            out2, embedding2 = model(data2)
            if args.use_out:#使用最后一层计算余弦相似度
                sim_out = F.cosine_similarity(out1, out2).cpu().data.numpy()
                similarities_out.append(sim_out)
            if args.use_embedding:#使用倒数第二层计算余弦相似度
                sim_embedding = F.cosine_similarity(embedding1, embedding2).cpu().data.numpy()
                similarities_embedding.append(sim_embedding)

            progress_bar.set_description('Test: [{}/{} ({:3.3f}%)]'.format(
                batch_idx + 1, len(test_loader), 100. * (batch_idx + 1) / len(test_loader)))

    pairs = np.concatenate(pairs)
    if args.use_out:
        similarities_out = np.array([sub_sim for sim in similarities_out for sub_sim in sim])
        if VAD:
            csv_file = 'pred_out_vad.csv'
        else:
            csv_file = 'pred_out.csv'
        with open(args.final_dir + csv_file, mode='w') as f:
            f.write('pairID,pred\n')
            for i in range(len(similarities_out)):
                f.write('{},{}\n'.format(pairs[i], similarities_out[i]))
    if args.use_embedding:
        similarities_embedding = np.array([sub_sim for sim in similarities_embedding for sub_sim in sim])
        if VAD:
            csv_file = 'pred_vad.csv'
        else:
            csv_file = 'pred.csv'
        with open(args.final_dir + csv_file, mode='w') as f:
            f.write('pairID,pred\n')
            for i in range(len(similarities_embedding)):
                f.write('{},{}\n'.format(pairs[i], similarities_embedding[i]))


def main():
    torch.manual_seed(args.seed)#设置随机种子

    train_dataset = SpeakerTrainDataset(samples_per_speaker=args.samples_per_speaker)#设置训练集读取
    n_classes = train_dataset.n_classes#说话人数
    print('Num of classes: {}'.format(n_classes))

    model = ResNet(layers=[1, 1, 1, 1], embedding_size=args.embedding_size, n_classes=n_classes, m=args.m).to(device)
    if args.optimizer == 'sgd':#优化器使用sgd
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, dampening=args.dampening, weight_decay=args.wd)
    elif args.optimizer == 'adagrad':#优化器使用adagrad
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, lr_decay=args.lr_decay, weight_decay=args.wd)
    else:#优化器使用adam
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = AngleLoss(lambda_min=args.lambda_min, lambda_max=args.lambda_max).to(device)#loss设置

    start = 1
    if args.resume:#是否从之前保存的模型开始
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.start is not None:
                start = start
            else:
                start = checkpoint['epoch'] + 1
            if args.load_it:
                criterion.it = checkpoint['it']
            elif args.it is not None:
                criterion.it = args.it
            if args.load_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if args.reset_linear:#迁移学习，需要重置最后一层
                state_dict = dict()
                for parameter in model.state_dict():
                    if parameter.find('linear') == -1:
                        state_dict[parameter] = checkpoint['state_dict'][parameter]
                    else:
                        state_dict[parameter] = model.state_dict()[parameter]
                checkpoint['state_dict'] = state_dict
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=1, pin_memory=True)
    for epoch in range(start, args.epochs + 1):
        train(epoch, model, criterion, optimizer, train_loader)

    test_dataset = SpeakerTestDataset(transform=args.transform)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             num_workers=1, pin_memory=True)
    test(model, test_loader)#测试