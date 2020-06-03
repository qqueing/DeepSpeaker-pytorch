#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: LossFunction.py
@Time: 2020/1/8 3:46 PM
@Overview:
"""
import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, alpha=10., partion=0.9):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.partion = partion

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.centers.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # 初始化权重，在第一维度上做normalize

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        norms = self.centers.data.norm(p=2, dim=1, keepdim=True).add(1e-14)
        self.centers.data = self.centers.data / norms * self.alpha

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

        # Variance for centers
        # variance = torch.std(self.centers, dim=1).sum()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        dist = dist.sum(dim=1).add(1e-14).sqrt()
        dist = dist.index_select(0, torch.argsort(dist)[-int(self.partion * batch_size):])
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / int(self.partion * batch_size)

        return loss

class TupleLoss(nn.Module):

    def __init__(self, batch_size, tuple_size):
        super(TupleLoss, self).__init__()
        self.batch_size = batch_size
        self.tuple_size = tuple_size
        self.sim = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, spk_representation, labels):
        """
        Args:
            x: (bashsize*tuplesize, dimension of linear layer)
            labels: ground truth labels with shape (batch_size).
        """
        feature_size = spk_representation.shape[1]
        w = torch.reshape(spk_representation, [self.batch_size, self.tuple_size, feature_size])

        loss = 0
        for indice_bash in range(self.batch_size):
            wi_enroll = w[indice_bash, 1:]  # shape:  (tuple_size-1, feature_size)
            wi_eval = w[indice_bash, 0]
            c_k = torch.mean(wi_enroll, dim=0)  # shape: (feature_size)
            # norm_c_k = c_k / torch.norm(c_k, p=2, keepdim=True)
            # normlize_ck = torch.norm(c_k, p=2, dim=0)
            # normlize_wi_eval = torch.norm(wi_eval, p=2, dim=0)
            cos_similarity = self.sim(c_k, wi_eval)
            score = cos_similarity

            loss += torch.sigmoid(score) * labels[indice_bash] + \
                    (1 - torch.sigmoid(score)*(1 - labels[indice_bash]))

        return -torch.log(loss / self.batch_size)
