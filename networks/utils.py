import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.autograd.variable as Variable
import numpy as np
import scipy.io as sio
from os.path import join as pjoin
#from skimage.transform import resize
#from models import HiFi1Edge
import skimage.io as io
import time
import skimage
import warnings
from PIL import Image

class Logger(object):
  def __init__(self, fpath=None):
    self.console = sys.stdout
    self.file = None
    if fpath is not None:
      self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
        self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
        self.file.flush()
        os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
        self.file.close()

class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_pretrained(model, fname, optimizer=None):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))

def load_vgg16pretrain(model, vggmodel='vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params =  model.state_dict()
    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size  == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)

def load_vgg16pretrain_half(model, vggmodel='vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params =  model.state_dict()
    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size  == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            shape = data.shape
            index = int(shape[0]/2)
            if len(shape) == 1:
                data = data[:index]
            else:
                data = data[:index,:,:,:]
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)

def load_fsds_caffe(model, fsdsmodel='caffe-fsds.mat'):
    fsds = sio.loadmat(fsdsmodel)
    torch_params =  model.state_dict()
    for k in fsds.keys():
        name_par = k.split('-')
        #print (name_par)
        size = len(name_par)

        data = np.squeeze(fsds[k])


        if 'upsample' in name_par:
           # print('skip upsample')
            continue


        if size  == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(fsds[k])
            if data.ndim==2:
                data = np.reshape(data, (data.shape[0], data.shape[1]))

            torch_params[name_space] = torch.from_numpy(data)

        if size  == 3:
           # if 'bias' in name_par:
            #    continue

            name_space = name_par[0] + '_' + name_par[1]+ '.' + name_par[2]
            data = np.squeeze(fsds[k])
           # print(data.shape)
            if data.ndim==2:
               # print (data.shape[0])
                data = np.reshape(data,(data.shape[0], data.shape[1]))
            if data.ndim==1 :
                data = np.reshape(data, (1, len(data), 1, 1))
            if data.ndim==0:
                data = np.reshape(data, (1))

            torch_params[name_space] = torch.from_numpy(data)

        if size == 4:
           # if 'bias' in name_par:
            #    continue
            data = np.squeeze(fsds[k])
            name_space = name_par[0] + '_' + name_par[1] + name_par[2] + '.' + name_par[3]
            if data.ndim==2:
                data = np.reshape(data,(data.shape[0], data.shape[1], 1, 1))

            torch_params[name_space] = torch.from_numpy(data)

    model.load_state_dict(torch_params)
    print('loaded')

def sgd_schedule(net, lr = 1e-6, momentum = 0.9, weight_decay = 2e-4):
    net_parameters_id = {}
    for pname, p in net.named_parameters():
        if pname in ['conv1_1.weight','conv1_2.weight',
                     'conv2_1.weight','conv2_2.weight',
                     'conv3_1.weight','conv3_2.weight','conv3_3.weight',
                     'conv4_1.weight','conv4_2.weight','conv4_3.weight',
                     'norm1_1.g','norm1_2.g',
                     'norm2_1.g','norm2_2.g',
                     'norm3_1.g','norm3_2.g','norm3_3.g',
                     'norm4_1.g','norm4_2.g','norm4_3.g']:
            print(pname, 'lr:1 de:1')
            if 'conv1-4.weight' not in net_parameters_id:
                net_parameters_id['conv1-4.weight'] = []
            net_parameters_id['conv1-4.weight'].append(p)
        elif pname in ['conv1_1.bias','conv1_2.bias',
                       'conv2_1.bias','conv2_2.bias',
                       'conv3_1.bias','conv3_2.bias','conv3_3.bias',
                       'conv4_1.bias','conv4_2.bias','conv4_3.bias',
                       'norm1_1.b','norm1_2.b',
                       'norm2_1.b','norm2_2.b',
                       'norm3_1.b','norm3_2.b','norm3_3.b',
                       'norm4_1.b','norm4_2.b','norm4_3.b']:
            print(pname, 'lr:2 de:0')
            if 'conv1-4.bias' not in net_parameters_id:
                net_parameters_id['conv1-4.bias'] = []
            net_parameters_id['conv1-4.bias'].append(p)
        elif pname in ['conv5_1.weight','conv5_2.weight','conv5_3.weight',
                       'norm5_1.g','norm5_2.g','norm5_3.g']:
            print(pname, 'lr:100 de:1')
            if 'conv5.weight' not in net_parameters_id:
                net_parameters_id['conv5.weight'] = []
            net_parameters_id['conv5.weight'].append(p)
        elif pname in ['conv5_1.bias','conv5_2.bias','conv5_3.bias',
                       'norm5_1.b','norm5_2.b','norm5_3.b'] :
            print(pname, 'lr:200 de:0')
            if 'conv5.bias' not in net_parameters_id:
                net_parameters_id['conv5.bias'] = []
            net_parameters_id['conv5.bias'].append(p)
        elif pname in ['conv1_1_down.weight','conv1_2_down.weight',
                       'conv2_1_down.weight','conv2_2_down.weight',
                       'conv3_1_down.weight','conv3_2_down.weight','conv3_3_down.weight',
                       'conv4_1_down.weight','conv4_2_down.weight','conv4_3_down.weight',
                       'conv5_1_down.weight','conv5_2_down.weight','conv5_3_down.weight']:
            # print(pname, 'lr:0.1 de:1')
            if 'conv_down_1-5.weight' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.weight'] = []
            net_parameters_id['conv_down_1-5.weight'].append(p)
        elif pname in ['conv1_1_down.bias','conv1_2_down.bias',
                       'conv2_1_down.bias','conv2_2_down.bias',
                       'conv3_1_down.bias','conv3_2_down.bias','conv3_3_down.bias',
                       'conv4_1_down.bias','conv4_2_down.bias','conv4_3_down.bias',
                       'conv5_1_down.bias','conv5_2_down.bias','conv5_3_down.bias']:
            # print(pname, 'lr:0.2 de:0')
            if 'conv_down_1-5.bias' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.bias'] = []
            net_parameters_id['conv_down_1-5.bias'].append(p)
        elif pname in ['score_dsn1.weight','score_dsn2.weight','score_dsn3.weight',
                       'score_dsn4.weight','score_dsn5.weight',
                       'atten1.weight','atten2.weight','atten3.weight',
                       'atten4.weight','atten5.weight',]:
            # print(pname, 'lr:0.01 de:1')
            if 'score_dsn_1-5.weight' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.weight'] = []
            net_parameters_id['score_dsn_1-5.weight'].append(p)
        elif pname in ['score_dsn1.bias','score_dsn2.bias','score_dsn3.bias',
                       'score_dsn4.bias','score_dsn5.bias',
                       'atten1.bias','atten2.bias','atten3.bias',
                       'atten4.bias','atten5.bias',]:
            # print(pname, 'lr:0.02 de:0')
            if 'score_dsn_1-5.bias' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.bias'] = []
            net_parameters_id['score_dsn_1-5.bias'].append(p)
        else:
            assert False

    optimizer = torch.optim.SGD([
            {'params': net_parameters_id['conv1-4.weight']      , 'lr': lr*1    , 'weight_decay': weight_decay},
            {'params': net_parameters_id['conv1-4.bias']        , 'lr': lr*2    , 'weight_decay': 0.},
            {'params': net_parameters_id['conv5.weight']        , 'lr': lr*100  , 'weight_decay': weight_decay},
            {'params': net_parameters_id['conv5.bias']          , 'lr': lr*200  , 'weight_decay': 0.},
            {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': lr*0.1  , 'weight_decay': weight_decay},
            {'params': net_parameters_id['conv_down_1-5.bias']  , 'lr': lr*0.2  , 'weight_decay': 0.},
            {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': lr*0.01 , 'weight_decay': weight_decay},
            {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': lr*0.02 , 'weight_decay': 0.},
        ], lr=lr, momentum=momentum, weight_decay= weight_decay)

    return optimizer



def adam_schedule(net, lr = 1e-6):
    net_parameters_id = {}
    for pname, p in net.named_parameters():
        if (not pname.endswith(".bias")) and ('conv1' in pname or 'conv2' in pname or 'conv3' in pname or 'conv4' in pname):
            print(pname, 'lr:1')
            if 'conv1-4' not in net_parameters_id:
                net_parameters_id['conv1-4'] = []
            net_parameters_id['conv1-4'].append(p)
        elif (not pname.endswith(".bias")) and ('conv5' in pname):
            print(pname, 'lr:10')
            if 'conv5' not in net_parameters_id:
                net_parameters_id['conv5'] = []
            net_parameters_id['conv5'].append(p)
        elif (not pname.endswith(".bias")) and ('convd' in pname):
            print(pname, 'lr:1')
            if 'convd1-5' not in net_parameters_id:
                net_parameters_id['convd1-5'] = []
            net_parameters_id['convd1-5'].append(p)
        elif (not pname.endswith(".b")) and ('norm1' in pname or 'norm2' in pname or 'norm3' in pname or 'norm4' in pname):
            print(pname, 'lr:0.1')
            if 'norm1-4' not in net_parameters_id:
                net_parameters_id['norm1-4'] = []
            net_parameters_id['norm1-4'].append(p)
        elif (not pname.endswith(".b")) and ('norm5' in pname):
            print(pname, 'lr:0.1')
            if 'norm5' not in net_parameters_id:
                net_parameters_id['norm5'] = []
            net_parameters_id['norm5'].append(p)
        elif (not pname.endswith(".b")) and ('normd' in pname):
            print(pname, 'lr:0.1')
            if 'normd' not in net_parameters_id:
                net_parameters_id['normd'] = []
            net_parameters_id['normd'].append(p)
        elif (not pname.endswith(".bias")) and ('score' in pname):
            print(pname, 'lr:1')
            if 'score_dsn_1-5' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5'] = []
            net_parameters_id['score_dsn_1-5'].append(p)
        elif (not pname.endswith(".bias")) and ('atten' in pname):
            print(pname, 'lr:1')
            if 'atten_1-5' not in net_parameters_id:
                net_parameters_id['atten_1-5'] = []
            net_parameters_id['atten_1-5'].append(p)
        
        elif ( pname.endswith(".bias")) and ('conv1' in pname or 'conv2' in pname or 'conv3' in pname or 'conv4' in pname):
            print(pname, 'lr:1')
            if 'conv1-4b' not in net_parameters_id:
                net_parameters_id['conv1-4b'] = []
            net_parameters_id['conv1-4b'].append(p)
        elif ( pname.endswith(".bias")) and ('conv5' in pname):
            print(pname, 'lr:10')
            if 'conv5b' not in net_parameters_id:
                net_parameters_id['conv5b'] = []
            net_parameters_id['conv5b'].append(p)
        elif ( pname.endswith(".bias")) and ('convd' in pname):
            print(pname, 'lr:1')
            if 'convd1-5b' not in net_parameters_id:
                net_parameters_id['convd1-5b'] = []
            net_parameters_id['convd1-5b'].append(p)
        elif ( pname.endswith(".b")) and ('norm1' in pname or 'norm2' in pname or 'norm3' in pname or 'norm4' in pname):
            print(pname, 'lr:0.1')
            if 'norm1-4b' not in net_parameters_id:
                net_parameters_id['norm1-4b'] = []
            net_parameters_id['norm1-4b'].append(p)
        elif ( pname.endswith(".b")) and ('norm5' in pname):
            print(pname, 'lr:0.1')
            if 'norm5b' not in net_parameters_id:
                net_parameters_id['norm5b'] = []
            net_parameters_id['norm5b'].append(p)
        elif ( pname.endswith(".b")) and ('normd' in pname):
            print(pname, 'lr:0.1')
            if 'normdb' not in net_parameters_id:
                net_parameters_id['normdb'] = []
            net_parameters_id['normdb'].append(p)
        elif ( pname.endswith(".bias")) and ('score' in pname):
            print(pname, 'lr:1')
            if 'score_dsn_1-5b' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5b'] = []
            net_parameters_id['score_dsn_1-5b'].append(p)
        elif ( pname.endswith(".bias")) and ('atten' in pname):
            print(pname, 'lr:1')
            if 'atten_1-5b' not in net_parameters_id:
                net_parameters_id['atten_1-5b'] = []
            net_parameters_id['atten_1-5b'].append(p)
        else:    
            assert False

    optimizer = torch.optim.AdamW([
            {'params': net_parameters_id['conv1-4']      , 'lr': lr*1  },
            {'params': net_parameters_id['conv5']        , 'lr': lr*100},
            {'params': net_parameters_id['norm1-4']      , 'lr': lr*1  },
            {'params': net_parameters_id['norm5']      , 'lr': lr*100  },
            #{'params': net_parameters_id['convd1-5']        , 'lr': lr*1},
            {'params': net_parameters_id['normd']      , 'lr': lr*1  },
            {'params': net_parameters_id['score_dsn_1-5'], 'lr': lr*1},
            {'params': net_parameters_id['atten_1-5'], 'lr': lr*1},   
                     
            {'params': net_parameters_id['conv1-4b']      , 'lr': lr*1   ,'weight_decay': 0.},
            {'params': net_parameters_id['conv5b']        , 'lr': lr*100,'weight_decay': 0.},
            {'params': net_parameters_id['norm1-4b']      , 'lr': lr*1  ,'weight_decay': 0.},
            {'params': net_parameters_id['norm5b']      , 'lr': lr*100  ,'weight_decay': 0.},
            #{'params': net_parameters_id['convd1-5b']        , 'lr': lr*1,'weight_decay': 0.},
            {'params': net_parameters_id['normdb']      , 'lr': lr*1  ,'weight_decay': 0.},
            {'params': net_parameters_id['score_dsn_1-5b'], 'lr': lr*1,'weight_decay': 0.},
            {'params': net_parameters_id['atten_1-5b'], 'lr': lr*1,'weight_decay': 0.},
        ], lr=lr)

    return optimizer


'''
def adam_schedule(net, lr = 1e-6):
    net_parameters_id = {}
    for pname, p in net.named_parameters():
        if (not pname.endswith(".bias")) and ('conv1' in pname or 'conv2' in pname or 'conv3' in pname or 'conv4' in pname):
            print(pname, 'lr:1')
            if 'conv1-4' not in net_parameters_id:
                net_parameters_id['conv1-4'] = []
            net_parameters_id['conv1-4'].append(p)
        elif (not pname.endswith(".bias")) and ('conv5' in pname):
            print(pname, 'lr:10')
            if 'conv5' not in net_parameters_id:
                net_parameters_id['conv5'] = []
            net_parameters_id['conv5'].append(p)
        elif (not pname.endswith(".bias")) and ('convd' in pname):
            print(pname, 'lr:1')
            if 'convd1-5' not in net_parameters_id:
                net_parameters_id['convd1-5'] = []
            net_parameters_id['convd1-5'].append(p)
        elif (not pname.endswith(".b")) and ('norm1' in pname or 'norm2' in pname or 'norm3' in pname or 'norm4' in pname):
            print(pname, 'lr:0.1')
            if 'norm1-4' not in net_parameters_id:
                net_parameters_id['norm1-4'] = []
            net_parameters_id['norm1-4'].append(p)
        elif (not pname.endswith(".b")) and ('norm5' in pname):
            print(pname, 'lr:0.1')
            if 'norm5' not in net_parameters_id:
                net_parameters_id['norm5'] = []
            net_parameters_id['norm5'].append(p)
        elif (not pname.endswith(".b")) and ('normd' in pname):
            print(pname, 'lr:0.1')
            if 'normd' not in net_parameters_id:
                net_parameters_id['normd'] = []
            net_parameters_id['normd'].append(p)
        elif (not pname.endswith(".bias")) and ('score' in pname):
            print(pname, 'lr:1')
            if 'score_dsn_1-5' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5'] = []
            net_parameters_id['score_dsn_1-5'].append(p)
        elif (not pname.endswith(".bias")) and ('atten' in pname):
            print(pname, 'lr:1')
            if 'atten_1-5' not in net_parameters_id:
                net_parameters_id['atten_1-5'] = []
            net_parameters_id['atten_1-5'].append(p)
        
        elif ( pname.endswith(".bias")) and ('conv1' in pname or 'conv2' in pname or 'conv3' in pname or 'conv4' in pname):
            print(pname, 'lr:1')
            if 'conv1-4b' not in net_parameters_id:
                net_parameters_id['conv1-4b'] = []
            net_parameters_id['conv1-4b'].append(p)
        elif ( pname.endswith(".bias")) and ('conv5' in pname):
            print(pname, 'lr:10')
            if 'conv5b' not in net_parameters_id:
                net_parameters_id['conv5b'] = []
            net_parameters_id['conv5b'].append(p)
        elif ( pname.endswith(".bias")) and ('convd' in pname):
            print(pname, 'lr:1')
            if 'convd1-5b' not in net_parameters_id:
                net_parameters_id['convd1-5b'] = []
            net_parameters_id['convd1-5b'].append(p)
        elif ( pname.endswith(".b")) and ('norm1' in pname or 'norm2' in pname or 'norm3' in pname or 'norm4' in pname):
            print(pname, 'lr:0.1')
            if 'norm1-4b' not in net_parameters_id:
                net_parameters_id['norm1-4b'] = []
            net_parameters_id['norm1-4b'].append(p)
        elif ( pname.endswith(".b")) and ('norm5' in pname):
            print(pname, 'lr:0.1')
            if 'norm5b' not in net_parameters_id:
                net_parameters_id['norm5b'] = []
            net_parameters_id['norm5b'].append(p)
        elif ( pname.endswith(".b")) and ('normd' in pname):
            print(pname, 'lr:0.1')
            if 'normdb' not in net_parameters_id:
                net_parameters_id['normdb'] = []
            net_parameters_id['normdb'].append(p)
        elif ( pname.endswith(".bias")) and ('score' in pname):
            print(pname, 'lr:1')
            if 'score_dsn_1-5b' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5b'] = []
            net_parameters_id['score_dsn_1-5b'].append(p)
        elif ( pname.endswith(".bias")) and ('atten' in pname):
            print(pname, 'lr:1')
            if 'atten_1-5b' not in net_parameters_id:
                net_parameters_id['atten_1-5b'] = []
            net_parameters_id['atten_1-5b'].append(p)
        else:    
            assert False

    optimizer = torch.optim.AdamW([
            {'params': net_parameters_id['conv1-4']      , 'lr': lr*1  },
            {'params': net_parameters_id['conv5']        , 'lr': lr*100},
            {'params': net_parameters_id['norm1-4']      , 'lr': lr*1  },
            {'params': net_parameters_id['norm5']      , 'lr': lr*100  },
            {'params': net_parameters_id['convd1-5']        , 'lr': lr*1},
            {'params': net_parameters_id['normd']      , 'lr': lr*1  },
            {'params': net_parameters_id['score_dsn_1-5'], 'lr': lr*1},
            #{'params': net_parameters_id['atten_1-5'], 'lr': lr*1},   
                     
            {'params': net_parameters_id['conv1-4b']      , 'lr': lr*1   ,'weight_decay': 0.},
            {'params': net_parameters_id['conv5b']        , 'lr': lr*100,'weight_decay': 0.},
            {'params': net_parameters_id['norm1-4b']      , 'lr': lr*1  ,'weight_decay': 0.},
            {'params': net_parameters_id['norm5b']      , 'lr': lr*100  ,'weight_decay': 0.},
            {'params': net_parameters_id['convd1-5b']        , 'lr': lr*1,'weight_decay': 0.},
            {'params': net_parameters_id['normdb']      , 'lr': lr*1  ,'weight_decay': 0.},
            {'params': net_parameters_id['score_dsn_1-5b'], 'lr': lr*1,'weight_decay': 0.},
            #{'params': net_parameters_id['atten_1-5b'], 'lr': lr*1,'weight_decay': 0.},
        ], lr=lr)

    return optimizer
'''
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1,4,1,1]):
            torch.nn.init.constant_(m.weight, 0.25)
        if m.bias is not None:
            m.bias.data.zero_()
