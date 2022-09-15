import numpy as np
import torch
from scipy import special
from torch.optim.optimizer import Optimizer
from torch import lgamma, digamma
import math
from typing import Tuple



def label2mask(label, total):
    mask = torch.zeros(label.shape).cuda()
    num_positive = torch.sum(((label >= (total / 2)) & (label != (total + 1))).float()).float()
    num_negative = torch.sum((label < (total / 2)).float()).float()
    mask[label < (total / 2)] = num_positive / (num_positive + num_negative)  # attention the order
    mask[(label >= (total / 2)) & (label != (total + 1))] = num_negative / (num_positive + num_negative)
    return mask


def knowleg2mask(know):
    mask = torch.zeros(know.shape).cuda()
    num_positive = torch.sum((know >= 0.5).float()).float()
    num_negative = torch.sum((know < 0.5).float()).float()
    mask[know < 0.5] =  num_positive / (num_positive + num_negative)  # attention the order
    mask[know >= 0.5] = num_negative / (num_positive + num_negative)
    return mask



def beta_expected_loglikelihood(prediction, label, mask, total):
    assert label.shape[0] == 1
    a = prediction[:, 0:1, :, :]
    b = prediction[:, 1:, :, :]
    eps = 0
    cost = - (label * digamma(a + eps) + (total - label) * digamma(b + eps) - total * digamma(a + b + 2 * eps))
    cost = cost * mask
    return torch.sum(cost)


def beta_knowledge(prediction, knowledge):
    mask = knowleg2mask(knowledge)
    assert prediction.shape[0] == 1
    a = prediction[:, 0:1, :, :]
    b = prediction[:, 1:, :, :]
    eps = 1e-3
    cost = -(torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b) +
             (a - 1) * torch.log(knowledge + eps) + (b - 1) * torch.log(1 - knowledge + eps))
    cost = cost * mask
    return torch.sum(cost)


