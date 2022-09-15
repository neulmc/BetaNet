from matplotlib.pyplot import sca
from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np
import cv2
import math

def prepare_image_cv2(im):
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


class Beta_BSDS(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', model = 'RCF'): # fuz: use fuz, fuz add: use fuz label
        self.root = root
        self.split = split
        self.model = model
        if self.split == 'train':
            self.filelist = join(self.root, 'train_pair.lst')
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split(',')
            lb_file = lb_file.replace('\n','')
            total_num = np.float(lb_file.split('_N')[1].split('.png')[0])
            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = lb[np.newaxis, :, :]
            
            if self.model == 'PiDiNet':
                dists = np.ones([5, lb.shape[1], lb.shape[2]])
                for dis_idx in range(5):
                    dist = np.array(Image.open(join(self.root, img_file.replace('train', 'distill_pidinet').split('.png')[0] + '_' + str(dis_idx) + '.png')), dtype=np.float32) / 255.0
                    dists[dis_idx] = dist
            else:
                dists = np.ones([6, lb.shape[1], lb.shape[2]])
                for dis_idx in range(6):
                    if self.model == 'RCF':
                        dist = np.array(Image.open(join(self.root, img_file.replace('train', 'distill_rcf').split('.png')[0] + '_' + str(dis_idx) + '.png')), dtype=np.float32) / 255.0
                    elif self.model == 'HED':
                        dist = np.array(Image.open(join(self.root, img_file.replace('train', 'distill_hed').split('.png')[0] + '_' + str(dis_idx) + '.png')), dtype=np.float32) / 255.0
                    dists[dis_idx] = dist
            img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_cv2(img)
            return img, lb, total_num, dists
        else:
            img_file = self.filelist[index].rstrip()
            img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_cv2(img)
            gt = np.array(Image.open(join(self.root, img_file.replace('test','test_gt').replace('.jpg','.png'))), dtype=np.float32)
            return img, gt, img_file.split('/')[-1].split('.')[0]


