#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: SoftmaxLoss.py
@Time: 2019/8/5 下午5:29
@Overview:
"AngleLinear" and "AngleSoftmaxLoss" Fork from
https://github.com/woshildh/a-softmax_pytorch/blob/master/a_softmax.py.

"AMSoftmax" Fork from
https://github.com/CoinCheung/pytorch-loss/blob/master/amsoftmax.py

"AngularSoftmax" is completed based on the two loss.

"Center Loss" is based on https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__=["AngleLinear", "AngleSoftmaxLoss", "AngularSoftmax", "AMSoftmax"]

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
        # ww = w.renorm(2, 1, 1e-5).mul(1e5)#方向0上做normalize
        x_modulus = x.norm(p=2, dim=1, keepdim=True)
        w_modulus = self.weight.norm(p=2, dim=0, keepdim=True)

        # x_len = x.pow(2).sum(1).pow(0.5)
        # w_len = ww.pow(2).sum(0).pow(0.5)

        cos_theta = x.mm(self.weight)
        cos_theta = cos_theta / x_modulus / w_modulus
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

        cos_theta = cos_theta * w_modulus * x_modulus
        phi_theta = phi_theta * w_modulus * x_modulus
        output = [cos_theta, phi_theta]#返回/cos(/theta)和/phi(/theta)
        return output

class AngleSoftmaxLoss(nn.Module):
    def __init__(self, lambda_min=5.0, lambda_max=1500.0, gamma=0, it=0):
        super(AngleSoftmaxLoss, self).__init__()
        self.gamma = gamma
        self.it = it
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def forward(self, x, y):
        '''
        x:
            cos_x: [batch, classes_num]
            phi_x: [batch, classes_num]
        y:
            target: LongTensor,[batch]
        return:
            loss:scalar
        '''
        self.it += 1
        cos_theta, phi_theta = x #output包括上面的[cos_theta, phi_theta]
        y = y.view(-1, 1)

        index = cos_theta.data * 0.0
        index.scatter_(1, y.data.view(-1, 1), 1)#将label存成稀疏矩阵
        index = index.bool()
        index = Variable(index)

        # set lamb, change the rate of softmax and A-softmax
        lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.it))  # 动态调整lambda，来调整cos(\theta)和\phi(\theta)的比例
        output = cos_theta * 1.0
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + lamb)  # 减去目标\cos(\theta)的部分
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + lamb)  # 加上目标\phi(\theta)的部分

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, y)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()

        return loss

class AngularSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 num_classes,
                 m=4,
                 gamma=0):
        super(AngularSoftmax, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.m = m
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.W = torch.nn.Parameter(torch.randn(in_feats, num_classes), requires_grad=True).cuda()
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal(self.W, gain=1)

        self.cos_function = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x,
        ]

    def forward(self, x, label):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # pdb.set_trace()
        # w_norm = self.W.norm(p=2, dim=1, keepdim=True)
        w = self.W  # / w_norm #[batch, out_planes]

        x_modulus = x.norm(p=2, dim=1, keepdim=True)
        w_modulus = w.norm(p=2, dim=0, keepdim=True)

        # get w@x=||w||*||x||*cos(theta)
        # w = w.cuda()
        inner_wx = x.mm(w) # [batch,out_planes]
        cos_theta = (inner_wx / x_modulus) / w_modulus

        cos_theta = cos_theta.clamp(-1, 1)

        # get cos(m*theta)
        cos_m_theta = self.cos_function[self.m](cos_theta)
        theta = Variable(cos_theta.data.acos())

        #get k, theta is in [ k*pi/m , (k+1)*pi/m ]
        k = (self.m * theta / math.pi).floor()
        minus_one = k*0 - 1

        # get phi_theta = -1^k*cos(m*theta)-2*k
        phi_theta = (minus_one ** k) * cos_m_theta - 2 * k

        # get cos_x and phi_x
        # cos_x = cos(theta)*||x||
        # phi_x = phi(theta)*||x||
        cos_x = cos_theta * x_modulus * w_modulus
        phi_x = phi_theta * x_modulus * w_modulus

        target = label.unsqueeze(-1)

        # get one_hot mat
        index = cos_x.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.bool()
        index = Variable(index)

        # set lamb, change the rate of softmax and A-softmax
        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))

        # get a-softmax and softmax mat
        output = cos_x * 1
        # output[index] -= (cos_x[index] * 1.0 / (+self.lamb))
        # output[index] += (phi_x[index] * 1.0 / (self.lamb))
        output[index] -= cos_x[index]
        output[index] += phi_x[index]

        loss = self.ce(output, label)

        return loss


class AdditiveMarginLinear(nn.Module):
    def __init__(self, feat_dim, n_classes=1000, use_gpu=False):
        super(AdditiveMarginLinear, self).__init__()
        self.feat_dim = feat_dim
        self.W = torch.nn.Parameter(torch.randn(feat_dim, n_classes), requires_grad=True)
        if use_gpu:
            self.W.cuda()
        nn.init.xavier_normal(self.W, gain=1)

    def forward(self, x):
        # assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.feat_dim

        # pdb.set_trace()
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)

        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)

        costh = torch.mm(x_norm, w_norm)

        return costh


class AMSoftmaxLoss(nn.Module):
    def __init__(self, margin=0.3, s=15):
        super(AMSoftmaxLoss, self).__init__()
        self.s = s
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()

    def forward(self, costh, label):
        lb_view = label.view(-1, 1)

        if lb_view.is_cuda:
            lb_view = lb_view.cpu()

        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view.data, self.margin)

        if costh.is_cuda:
            delt_costh = Variable(delt_costh.cuda())

        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m

        loss = self.ce(costh_m_s, label)

        return loss

class AMSoftmax(nn.Module):
    def __init__(self, in_feats, n_classes=10, m=0.3, s=15, use_gpu=True):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()

        if use_gpu:
            self.W.cuda()
            self.ce = self.ce.cuda()

        nn.init.xavier_normal(self.W, gain=1)

    def forward(self, x, label):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # pdb.set_trace()
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)

        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)

        costh = torch.mm(x_norm, w_norm)

        lb_view = label.view(-1, 1)

        if lb_view.is_cuda:
            lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view.data, self.m)

        if x.is_cuda:
            delt_costh = Variable(delt_costh.cuda())

        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m

        loss = self.ce(costh_m_s, label)

        return costh, loss


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.centers.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # 初始化权重，在第一维度上做normalize

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        #if self.use_gpu: classes = classes.cuda()
        if self.centers.is_cuda:
            classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

# Testing those Loss Classes
# a = Variable(torch.Tensor([[1., 1., 3.],
#                   [1., 2., 0.],
#                   [1., 4., 3.],
#                   [5., 0., 3.]]).cuda())
#
# a_label = Variable(torch.LongTensor([2, 1, 1, 0]).cuda())
#
# linear = AngleLinear(in_planes=3, out_planes=3, m=4)
# Asoft = AngleSoftmaxLoss()
# a_linear = linear(a)
# a_asoft = Asoft(a_linear, a_label)
# print("axsoftmax loss is {}".format(a_asoft))
#
# asoft = AngularSoftmax(in_feats=3, num_classes=3)
# a_loss = asoft(a, a_label)
# print("amsoftmax loss is {}".format(a_loss))
#
# amsoft = AMSoftmax(in_feats=3)
# am_a = amsoft(a, a_label)
# print("amsoftmax loss is {}".format(am_a))