import os, sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd.variable as Variable
import numpy as np
import scipy.io as sio
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _single, _pair, _triple
import torch.nn.functional as F

from torch.autograd import Variable  # torch 中 Variable 模块
import math

# norm
class LayerNorm_conv(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


# main network
class RCF(nn.Module):
    def __init__(self, linear_dim=21):
        super(RCF, self).__init__()
        
        # lr 1 2 decay 1 0
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)

        self.normd1_1 = LayerNorm_conv(linear_dim)
        self.normd1_2 = LayerNorm_conv(linear_dim)
        self.normd2_1 = LayerNorm_conv(linear_dim)
        self.normd2_2 = LayerNorm_conv(linear_dim)
        self.normd3_1 = LayerNorm_conv(linear_dim)
        self.normd3_2 = LayerNorm_conv(linear_dim)
        self.normd3_3 = LayerNorm_conv(linear_dim)
        self.normd4_1 = LayerNorm_conv(linear_dim)
        self.normd4_2 = LayerNorm_conv(linear_dim)
        self.normd4_3 = LayerNorm_conv(linear_dim)
        self.normd5_1 = LayerNorm_conv(linear_dim)
        self.normd5_2 = LayerNorm_conv(linear_dim)
        self.normd5_3 = LayerNorm_conv(linear_dim)

        self.activ = nn.GELU()

        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        self.maxpool_1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool_2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool_3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # lr 0.1 0.2 decay 1 0

        self.conv1_1_down = nn.Conv2d(64, linear_dim, 3, padding=1)
        self.conv1_2_down = nn.Conv2d(64, linear_dim, 3, padding=1)

        self.conv2_1_down = nn.Conv2d(128, linear_dim, 3, padding=1)
        self.conv2_2_down = nn.Conv2d(128, linear_dim, 3, padding=1)

        self.conv3_1_down = nn.Conv2d(256, linear_dim, 3, padding=1)
        self.conv3_2_down = nn.Conv2d(256, linear_dim, 3, padding=1)
        self.conv3_3_down = nn.Conv2d(256, linear_dim, 3, padding=1)

        self.conv4_1_down = nn.Conv2d(512, linear_dim, 3, padding=1)
        self.conv4_2_down = nn.Conv2d(512, linear_dim, 3, padding=1)
        self.conv4_3_down = nn.Conv2d(512, linear_dim, 3, padding=1)

        self.conv5_1_down = nn.Conv2d(512, linear_dim, 3, padding=1)
        self.conv5_2_down = nn.Conv2d(512, linear_dim, 3, padding=1)
        self.conv5_3_down = nn.Conv2d(512, linear_dim, 3, padding=1)

        self.score_dsn1 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                                  LayerNorm_conv(linear_dim), self.activ,
                                                  nn.Conv2d(linear_dim, 2, 1))
        self.score_dsn2 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                                  LayerNorm_conv(linear_dim), self.activ,
                                                  nn.Conv2d(linear_dim, 2, 1))
        self.score_dsn3 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                                  LayerNorm_conv(linear_dim), self.activ,
                                                  nn.Conv2d(linear_dim, 2, 1))
        self.score_dsn4 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                                  LayerNorm_conv(linear_dim), self.activ,
                                                  nn.Conv2d(linear_dim, 2, 1))
        self.score_dsn5 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                                  LayerNorm_conv(linear_dim), self.activ,
                                                  nn.Conv2d(linear_dim, 2, 1))

        # lr 0.001 0.002 decay 1 0
        self.atten1 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                                  LayerNorm_conv(linear_dim), self.activ,
                                                  nn.Conv2d(linear_dim, 2, 1))
        self.atten2 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                                  LayerNorm_conv(linear_dim), self.activ,
                                                  nn.Conv2d(linear_dim, 2, 1))
        self.atten3 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                                  LayerNorm_conv(linear_dim), self.activ,
                                                  nn.Conv2d(linear_dim, 2, 1))
        self.atten4 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                                  LayerNorm_conv(linear_dim), self.activ,
                                                  nn.Conv2d(linear_dim, 2, 1))
        self.atten5 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                                  LayerNorm_conv(linear_dim), self.activ,
                                                  nn.Conv2d(linear_dim, 2, 1))

        self.weight_deconv2 = make_bilinear_weights(4, 2).cuda()
        self.weight_deconv3 = make_bilinear_weights(8, 2).cuda()
        self.weight_deconv4 = make_bilinear_weights(16, 2).cuda()
        self.weight_deconv5 = make_bilinear_weights(16, 2).cuda()


    def trans_map(self, x):
        ls = []
        for xidx, xi in enumerate(x):
            ls.append(torch.exp(torch.abs(xi)))  # abs
        return ls

    def expect_value(self, x):
        return [xi[:, 0, :, :] / (xi[:, 0, :, :] + xi[:, 1, :, :]) for xi in x]

    def forward(self, x, reture_expect=False):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        
        conv1_1 = self.activ(self.conv1_1(x))
        conv1_2 = self.activ(self.conv1_2(conv1_1))
        pool1 = self.maxpool_1(conv1_2)

        conv2_1 = self.activ(self.conv2_1(pool1))
        conv2_2 = self.activ(self.conv2_2(conv2_1))
        pool2 = self.maxpool_2(conv2_2)

        conv3_1 = self.activ(self.conv3_1(pool2))
        conv3_2 = self.activ(self.conv3_2(conv3_1))
        conv3_3 = self.activ(self.conv3_3(conv3_2))
        pool3 = self.maxpool_3(conv3_3)

        conv4_1 = self.activ(self.conv4_1(pool3))
        conv4_2 = self.activ(self.conv4_2(conv4_1))
        conv4_3 = self.activ(self.conv4_3(conv4_2))
        pool4 = self.maxpool4(conv4_3)

        conv5_1 = self.activ(self.conv5_1(pool4))
        conv5_2 = self.activ(self.conv5_2(conv5_1))
        conv5_3 = self.activ(self.conv5_3(conv5_2))

        conv1_1_down = self.activ(self.normd1_1(self.conv1_1_down(conv1_1)))
        conv1_2_down = self.activ(self.normd1_2(self.conv1_2_down(conv1_2)))
        conv2_1_down = self.activ(self.normd2_1(self.conv2_1_down(conv2_1)))
        conv2_2_down = self.activ(self.normd2_2(self.conv2_2_down(conv2_2)))
        conv3_1_down = self.activ(self.normd3_1(self.conv3_1_down(conv3_1)))
        conv3_2_down = self.activ(self.normd3_2(self.conv3_2_down(conv3_2)))
        conv3_3_down = self.activ(self.normd3_3(self.conv3_3_down(conv3_3)))
        conv4_1_down = self.activ(self.normd4_1(self.conv4_1_down(conv4_1)))
        conv4_2_down = self.activ(self.normd4_2(self.conv4_2_down(conv4_2)))
        conv4_3_down = self.activ(self.normd4_3(self.conv4_3_down(conv4_3)))
        conv5_1_down = self.activ(self.normd5_1(self.conv5_1_down(conv5_1)))
        conv5_2_down = self.activ(self.normd5_2(self.conv5_2_down(conv5_2)))
        conv5_3_down = self.activ(self.normd5_3(self.conv5_3_down(conv5_3)))

        so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, self.weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, self.weight_deconv5, stride=8)

        ### center crop
        so1 = so1_out
        so2 = crop_center(upsample2, img_H, img_W)
        so3 = crop_center(upsample3, img_H, img_W)
        so4 = crop_center(upsample4, img_H, img_W)
        so5 = crop_center(upsample5, img_H, img_W)

        at1_out = self.atten1(conv1_1_down + conv1_2_down)
        at2_out = self.atten2(conv2_1_down + conv2_2_down)
        at3_out = self.atten3(conv3_1_down + conv3_2_down + conv3_3_down)
        at4_out = self.atten4(conv4_1_down + conv4_2_down + conv4_3_down)
        at5_out = self.atten5(conv5_1_down + conv5_2_down + conv5_3_down)

        at_upsample2 = torch.nn.functional.conv_transpose2d(at2_out, self.weight_deconv2, stride=2)
        at_upsample3 = torch.nn.functional.conv_transpose2d(at3_out, self.weight_deconv3, stride=4)
        at_upsample4 = torch.nn.functional.conv_transpose2d(at4_out, self.weight_deconv4, stride=8)
        at_upsample5 = torch.nn.functional.conv_transpose2d(at5_out, self.weight_deconv5, stride=8)

        ### center crop
        at1 = at1_out
        at2 = crop_center(at_upsample2, img_H, img_W)
        at3 = crop_center(at_upsample3, img_H, img_W)
        at4 = crop_center(at_upsample4, img_H, img_W)
        at5 = crop_center(at_upsample5, img_H, img_W)

        fusecat_a = torch.cat((so1[:, 0:1], so2[:, 0:1], so3[:, 0:1], so4[:, 0:1], so5[:, 0:1]), dim=1)
        fusecat_b = torch.cat((so1[:, 1:], so2[:, 1:], so3[:, 1:], so4[:, 1:], so5[:, 1:]), dim=1)
        at_fusecat_a = torch.cat((at1[:, 0:1], at2[:, 0:1], at3[:, 0:1], at4[:, 0:1], at5[:, 0:1]), dim=1)
        at_fusecat_b = torch.cat((at1[:, 1:], at2[:, 1:], at3[:, 1:], at4[:, 1:], at5[:, 1:]), dim=1)
        soft_p = torch.softmax(at_fusecat_a, dim=1)
        soft_n = torch.softmax(at_fusecat_b, dim=1)

        soft_p = soft_p / 2 + 0.1
        soft_n = soft_n / 2 + 0.1

        self.soft_p_tmp = torch.mean(soft_p, dim=(2, 3))
        self.soft_n_tmp = torch.mean(soft_n, dim=(2, 3))
        fuse_a = torch.sum(torch.abs(fusecat_a) * soft_p, dim=1, keepdim=True)
        fuse_b = torch.sum(torch.abs(fusecat_b) * soft_n, dim=1, keepdim=True)
  
        results = [so1, so2, so3, so4, so5, torch.cat((fuse_a, fuse_b), dim=1)]
        results = self.trans_map(results)

        if reture_expect:
            results = self.expect_value(results)
        return results

class HED(nn.Module):
    def __init__(self, linear_dim = 21):
        super(HED, self).__init__()
        # lr 1 2 decay 1 0
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.normd1_2 = LayerNorm_conv(linear_dim)
        self.normd2_2 = LayerNorm_conv(linear_dim)
        self.normd3_3 = LayerNorm_conv(linear_dim)
        self.normd4_3 = LayerNorm_conv(linear_dim)
        self.normd5_3 = LayerNorm_conv(linear_dim)

        self.activ = nn.GELU()

        self.maxpool_4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool_1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool_2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool_3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv1_2_down = nn.Conv2d(64, linear_dim, 3, padding=1)
        self.conv2_2_down = nn.Conv2d(128, linear_dim, 3, padding=1)
        self.conv3_3_down = nn.Conv2d(256, linear_dim, 3, padding=1)
        self.conv4_3_down = nn.Conv2d(512, linear_dim, 3, padding=1)
        self.conv5_3_down = nn.Conv2d(512, linear_dim, 3, padding=1)

        self.score_dsn1 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                            LayerNorm_conv(linear_dim), self.activ,
                                            nn.Conv2d(linear_dim, 2, 1))
        self.score_dsn2 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                            LayerNorm_conv(linear_dim), self.activ,
                                            nn.Conv2d(linear_dim, 2, 1))
        self.score_dsn3 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                            LayerNorm_conv(linear_dim), self.activ,
                                            nn.Conv2d(linear_dim, 2, 1))
        self.score_dsn4 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                            LayerNorm_conv(linear_dim), self.activ,
                                            nn.Conv2d(linear_dim, 2, 1))
        self.score_dsn5 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                            LayerNorm_conv(linear_dim), self.activ,
                                            nn.Conv2d(linear_dim, 2, 1))

        self.atten1 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                        LayerNorm_conv(linear_dim), self.activ,
                                        nn.Conv2d(linear_dim, 2, 1))
        self.atten2 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                        LayerNorm_conv(linear_dim), self.activ,
                                        nn.Conv2d(linear_dim, 2, 1))
        self.atten3 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                        LayerNorm_conv(linear_dim), self.activ,
                                        nn.Conv2d(linear_dim, 2, 1))
        self.atten4 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                        LayerNorm_conv(linear_dim), self.activ,
                                         nn.Conv2d(linear_dim, 2, 1))
        self.atten5 = torch.nn.Sequential(nn.Conv2d(linear_dim, linear_dim, 3, padding=1),
                                        LayerNorm_conv(linear_dim), self.activ,
                                        nn.Conv2d(linear_dim, 2, 1))

        ## Fixed the upsampling weights for the training process as per @https://github.com/xwjabc/hed
        self.weight_deconv2 = make_bilinear_weights(4, 2).cuda()
        self.weight_deconv3 = make_bilinear_weights(8, 2).cuda()
        self.weight_deconv4 = make_bilinear_weights(16, 2).cuda()
        self.weight_deconv5 = make_bilinear_weights(32, 2).cuda()

    def trans_map(self, x):
        ls = []
        for xidx, xi in enumerate(x):
            ls.append(torch.exp(torch.abs(xi)))  # abs
        return ls

    def expect_value(self, x):
        return [xi[:, 0, :, :] / (xi[:, 0, :, :] + xi[:, 1, :, :]) for xi in x]

    def forward(self, x, reture_expect=False):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]

        conv1_1 = self.activ(self.conv1_1(x))
        conv1_2 = self.activ(self.conv1_2(conv1_1))
        pool1 = self.maxpool_1(conv1_2)

        conv2_1 = self.activ(self.conv2_1(pool1))
        conv2_2 = self.activ(self.conv2_2(conv2_1))
        pool2 = self.maxpool_2(conv2_2)

        conv3_1 = self.activ(self.conv3_1(pool2))
        conv3_2 = self.activ(self.conv3_2(conv3_1))
        conv3_3 = self.activ(self.conv3_3(conv3_2))
        pool3 = self.maxpool_3(conv3_3)

        conv4_1 = self.activ(self.conv4_1(pool3))
        conv4_2 = self.activ(self.conv4_2(conv4_1))
        conv4_3 = self.activ(self.conv4_3(conv4_2))
        pool4 = self.maxpool_4(conv4_3)

        conv5_1 = self.activ(self.conv5_1(pool4))
        conv5_2 = self.activ(self.conv5_2(conv5_1))
        conv5_3 = self.activ(self.conv5_3(conv5_2))

        conv1_2_down = self.activ(self.normd1_2(self.conv1_2_down(conv1_2)))
        conv2_2_down = self.activ(self.normd2_2(self.conv2_2_down(conv2_2)))
        conv3_3_down = self.activ(self.normd3_3(self.conv3_3_down(conv3_3)))
        conv4_3_down = self.activ(self.normd4_3(self.conv4_3_down(conv4_3)))
        conv5_3_down = self.activ(self.normd5_3(self.conv5_3_down(conv5_3)))

        so1_out = self.score_dsn1(conv1_2_down)
        so2_out = self.score_dsn2(conv2_2_down)
        so3_out = self.score_dsn3(conv3_3_down)
        so4_out = self.score_dsn4(conv4_3_down)
        so5_out = self.score_dsn5(conv5_3_down)

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, self.weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, self.weight_deconv5, stride=16)

        ### center crop
        so1 = so1_out
        so2 = crop_center(upsample2, img_H, img_W)
        so3 = crop_center(upsample3, img_H, img_W)
        so4 = crop_center(upsample4, img_H, img_W)
        so5 = crop_center(upsample5, img_H, img_W)

        at1_out = self.atten1(conv1_2_down)
        at2_out = self.atten2(conv2_2_down)
        at3_out = self.atten3(conv3_3_down)
        at4_out = self.atten4(conv4_3_down)
        at5_out = self.atten5(conv5_3_down)

        at_upsample2 = torch.nn.functional.conv_transpose2d(at2_out, self.weight_deconv2, stride=2)
        at_upsample3 = torch.nn.functional.conv_transpose2d(at3_out, self.weight_deconv3, stride=4)
        at_upsample4 = torch.nn.functional.conv_transpose2d(at4_out, self.weight_deconv4, stride=8)
        at_upsample5 = torch.nn.functional.conv_transpose2d(at5_out, self.weight_deconv5, stride=16)

        ### center crop
        at1 = at1_out
        at2 = crop_center(at_upsample2, img_H, img_W)
        at3 = crop_center(at_upsample3, img_H, img_W)
        at4 = crop_center(at_upsample4, img_H, img_W)
        at5 = crop_center(at_upsample5, img_H, img_W)

        fusecat_a = torch.cat((so1[:, 0:1], so2[:, 0:1], so3[:, 0:1], so4[:, 0:1], so5[:, 0:1]), dim=1)
        fusecat_b = torch.cat((so1[:, 1:], so2[:, 1:], so3[:, 1:], so4[:, 1:], so5[:, 1:]), dim=1)
        at_fusecat_a = torch.cat((at1[:, 0:1], at2[:, 0:1], at3[:, 0:1], at4[:, 0:1], at5[:, 0:1]), dim=1)
        at_fusecat_b = torch.cat((at1[:, 1:], at2[:, 1:], at3[:, 1:], at4[:, 1:], at5[:, 1:]), dim=1)
        soft_p = torch.softmax(at_fusecat_a, dim=1)
        soft_n = torch.softmax(at_fusecat_b, dim=1)
        # lmc
        soft_p = soft_p / 2 + 0.1
        soft_n = soft_n / 2 + 0.1

        self.soft_p_tmp = torch.mean(soft_p, dim=(2, 3))
        self.soft_n_tmp = torch.mean(soft_n, dim=(2, 3))
        fuse_a = torch.sum(torch.abs(fusecat_a) * soft_p, dim=1, keepdim=True)
        fuse_b = torch.sum(torch.abs(fusecat_b) * soft_n, dim=1, keepdim=True)
        results = [so1, so2, so3, so4, so5, torch.cat((fuse_a, fuse_b), dim=1)]
        results = self.trans_map(results)

        if reture_expect:
            results = self.expect_value(results)
        return results

class PiDiNet(nn.Module):
    def __init__(self, inplane = 60, dil = 24):
        super(PiDiNet, self).__init__()
        pdcs = config_model()
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        self.fuseplanes = []

        self.inplane = inplane

        self.init_block = Conv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)
        block_class = PDCBlock

        self.block1_1 = block_class(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = block_class(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = block_class(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 2C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = block_class(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = block_class(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = block_class(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = block_class(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        self.block4_1 = block_class(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = block_class(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = block_class(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = block_class(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        self.dilations = nn.ModuleList()
        self.conv_reduces = nn.ModuleList()
        self.attens = nn.ModuleList()

        for i in range(4):
            self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
            self.conv_reduces.append(MapReduce(self.dil))
            self.attens.append(MapReduce(self.dil))

    def trans_map(self, x):
        ls = []
        for xidx, xi in enumerate(x):
            ls.append(torch.exp(torch.abs(xi)))  # abs
        return ls

    def expect_value(self, x):
        return [xi[:, 0, :, :] / (xi[:, 0, :, :] + xi[:, 1, :, :]) for xi in x]

    def forward(self, x, reture_expect = False):
        H, W = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_fuses = []

        for i, xi in enumerate([x1, x2, x3, x4]):
            x_fuses.append(self.dilations[i](xi))

        so1 = self.conv_reduces[0](x_fuses[0])
        so1 = F.interpolate(so1, (H, W), mode="bilinear", align_corners=False)
        so2 = self.conv_reduces[1](x_fuses[1])
        so2 = F.interpolate(so2, (H, W), mode="bilinear", align_corners=False)
        so3 = self.conv_reduces[2](x_fuses[2])
        so3 = F.interpolate(so3, (H, W), mode="bilinear", align_corners=False)
        so4 = self.conv_reduces[3](x_fuses[3])
        so4 = F.interpolate(so4, (H, W), mode="bilinear", align_corners=False)

        at1 = self.attens[0](x_fuses[0])
        at1 = F.interpolate(at1, (H, W), mode="bilinear", align_corners=False)
        at2 = self.attens[1](x_fuses[1])
        at2 = F.interpolate(at2, (H, W), mode="bilinear", align_corners=False)
        at3 = self.attens[2](x_fuses[2])
        at3 = F.interpolate(at3, (H, W), mode="bilinear", align_corners=False)
        at4 = self.attens[3](x_fuses[3])
        at4 = F.interpolate(at4, (H, W), mode="bilinear", align_corners=False)

        fusecat_a = torch.cat((so1[:, 0:1], so2[:, 0:1], so3[:, 0:1], so4[:, 0:1]), dim=1)
        fusecat_b = torch.cat((so1[:, 1:], so2[:, 1:], so3[:, 1:], so4[:, 1:]), dim=1)
        at_fusecat_a = torch.cat((at1[:, 0:1], at2[:, 0:1], at3[:, 0:1], at4[:, 0:1]), dim=1)
        at_fusecat_b = torch.cat((at1[:, 1:], at2[:, 1:], at3[:, 1:], at4[:, 1:]), dim=1)
        soft_p = torch.softmax(at_fusecat_a, dim=1)
        soft_n = torch.softmax(at_fusecat_b, dim=1)
        # lmc
        soft_p = soft_p / 2 + 0.125
        soft_n = soft_n / 2 + 0.125
        self.soft_p_tmp = torch.mean(soft_p, dim=(2, 3))
        self.soft_n_tmp = torch.mean(soft_n, dim=(2, 3))
        fuse_a = torch.sum(torch.abs(fusecat_a) * soft_p, dim=1, keepdim=True)
        fuse_b = torch.sum(torch.abs(fusecat_b) * soft_n, dim=1, keepdim=True)
        results = [so1, so2, so3, so4, torch.cat((fuse_a, fuse_b), dim=1)]
        results = self.trans_map(results)

        if reture_expect:
            results = self.expect_value(results)
        return results

# blocks 

class CDCM(nn.Module):
    """
    Compact Dilation Convolution based Module
    """

    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        self.norm2 = LayerNorm_conv(out_channels)
        self.act = nn.GELU()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return self.act(self.norm2(x1 + x2 + x3 + x4))

class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """

    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = LayerNorm_conv(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, 2, kernel_size=1, padding=0)

    def forward(self, x):
        return self.conv2(self.act(self.norm1(self.conv1(x))))

class PDCBlock(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock, self).__init__()
        self.stride = stride

        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.GELU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

class Conv2d(nn.Module):
    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def createConvFunc(op_type):
    assert op_type in ['cv', 'cd', 'ad', 'rd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape) # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = 0
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    else:
        print('impossible to be here unless you force that')
        return None

def config_model():
    nets = {'carv4': {
        'layer0':  'cd',
        'layer1':  'ad',
        'layer2':  'rd',
        'layer3':  'cv',
        'layer4':  'cd',
        'layer5':  'ad',
        'layer6':  'rd',
        'layer7':  'cv',
        'layer8':  'cd',
        'layer9':  'ad',
        'layer10': 'rd',
        'layer11': 'cv',
        'layer12': 'cd',
        'layer13': 'ad',
        'layer14': 'rd',
        'layer15': 'cv',
        }}

    pdcs = []
    for i in range(16):
        layer_name = 'layer%d' % i
        op = nets['carv4'][layer_name]
        pdcs.append(createConvFunc(op))

    return pdcs

# Based on BDCN Implementation @ https://github.com/pkuCactus/BDCN
def crop(data1, h, w, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    assert (h <= h1 and w <= w1)
    data = data1[:, :, crop_h:crop_h + h, crop_w:crop_w + w]
    return data


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


def crop_center(data1, h, w):
    _, _, h1, w1 = data1.size()
    assert (h <= h1 and w <= w1)
    crop_h = int((h1 - h) / 2)
    crop_w = int((w1 - w) / 2)
    data = data1[:, :, crop_h:crop_h + h, crop_w:crop_w + w]
    return data


def upsample(input, stride, num_channels=1):
    kernel_size = stride * 2
    kernel = make_bilinear_weights(kernel_size, num_channels).cuda()
    return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)
