import os, sys
from statistics import mode
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
import torchvision
from networks.data_loader import Beta_BSDS
from networks.models import RCF, HED, PiDiNet
from networks.functions import label2mask, beta_expected_loglikelihood, beta_knowledge
from torch.utils.data import DataLoader
from networks.utils import Logger, Averagvalue, save_checkpoint, load_vgg16pretrain
from os.path import join, isdir
import shutil
from torch.optim import lr_scheduler

# this file is used for analysing our mutil-scale knowledge distill
# data in/out
out_dir = 'beta_RCF_dist0'
dataset = '../BetaBSDS'
model_name = 'RCF' # RCF HED PiDiNet

# dis
dist_id = 0 # 0,1,2,3,4, or 5 which scale for knowledge distill

# loss
W_seq = 1
W_know = 0.05
prior_norm = 100

# lr parameters
lr = 1e-3
lr_stepsize = 5
lr_decay = 0.1

batch_size = 1
itersize = 10
maxepoch = 10
grad_clamp = 10

# log/environment
print_freq = 100
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
torch.set_num_threads(4)

# load
load_dir = None

# random
modelseed = 1
dataseed = 1

def main():
    print(os.path.realpath(__file__))
    
    # dataset
    train_dataset = Beta_BSDS(root=dataset, model = model_name, split="train")
    test_dataset = Beta_BSDS(root=dataset, split="test")
    torch.manual_seed(dataseed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=0, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,num_workers=0, drop_last=True, shuffle=False)

    # preparing
    TMP_DIR = 'experiences_revised/' + out_dir
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    if os.path.exists(TMP_DIR + '/networks'):
        shutil.rmtree(TMP_DIR + '/networks')
    shutil.copy(os.path.realpath(__file__), TMP_DIR + '/config')
    shutil.copytree('networks', TMP_DIR + '/networks')

    # model PiDiNet
    if model_name == 'RCF':
        model = RCF()
    elif model_name == 'HED':
        model = HED()
    elif model_name == 'PiDiNet':
        model = PiDiNet()
    else:
        assert False
    model.cuda()
    torch.manual_seed(modelseed)
    model.apply(weights_init)
    if load_dir is not None:
        model.load_state_dict(torch.load(load_dir)['state_dict'])
        print('load!')
    elif isinstance(model, RCF) or isinstance(model, HED):
        load_vgg16pretrain(model)

    # tune lr
    net = model
    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=lr_decay)
    
    # log
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' % ('adam', lr)))
    sys.stdout = log
    #test(model, test_loader, save_dir=join(TMP_DIR, 'epoch-%d-testing-record-view' % -1))

    train_loss = []
    train_loss_detail = []
    for epoch in range(0, maxepoch):
        print('lr_rate: %f' %optimizer.param_groups[0]['lr'])
        tr_avg_loss, tr_detail_loss = train(
            train_loader, model, optimizer, epoch,
            save_dir=join(TMP_DIR, 'epoch-%d-training-record' % epoch))
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, filename=join(TMP_DIR, "epoch-%d-checkpoint.pth" % epoch))

        test(model, test_loader, save_dir=join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))

        log.flush()  # write log
        # save train/val loss/accuracy, save every epoch in case of early stop
        train_loss.append(tr_avg_loss)
        train_loss_detail += tr_detail_loss
        scheduler.step() # will adjust learning rate
        
def train(train_loader, model, optimizer, epoch, save_dir):
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    losses_prior = Averagvalue()
    losses_seq = Averagvalue()
    losses_know = Averagvalue()
    # switch to train mode
    model.train()
    end = time.time()
    epoch_loss = []
    counter = 0

    for i, (image, label, total, dists) in enumerate(train_loader):
        data_time.update(time.time() - end)
        image, label, total, dists = image.cuda(), label.cuda(), total.cuda(), dists.cuda()
        mask = label2mask(label, total)
        outputs = model(image)
        loss_prior = torch.zeros(1).cuda()
        loss_seq = torch.zeros(1).cuda()
        loss_know = torch.zeros(1).cuda()
        regular_mn = torch.mean(outputs[0] + outputs[1] + outputs[2] + outputs[3] + outputs[4]).detach()
        for idxo, o in enumerate(outputs):
            loss_prior = loss_prior + torch.mean((prior_norm - torch.clamp(torch.sum(o, dim = 1), min = prior_norm) ) **2) / itersize       
            loss_seq = loss_seq + beta_expected_loglikelihood(o, label, mask, total) * W_seq / itersize
            loss_know = loss_know + beta_knowledge(o, dists[:, dist_id:dist_id + 1]) * W_know / itersize
        counter += 1
        loss = loss_seq + loss_know + loss_prior   

        loss.backward()
        if counter == itersize:
            if grad_clamp is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clamp)
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
                
        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        losses_prior.update(loss_prior.item(), image.size(0))
        losses_seq.update(loss_seq.item(), image.size(0))
        losses_know.update(loss_know.item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)
        if i % print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, maxepoch, i, len(train_loader)) + \
                    'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                    'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                    'LossS {losses_seq.val:f} (avg:{losses_seq.avg:f}) '.format(losses_seq=losses_seq) + \
                    'LossK {losses_know.val:f} (avg:{losses_know.avg:f}) '.format(losses_know=losses_know) + \
                    'LossP {losses_prior.val:f} (avg:{losses_prior.avg:f}) '.format(losses_prior=losses_prior)
            print(info)
            print(model.soft_p_tmp.detach().cpu().numpy()[0, :])
            print(model.soft_n_tmp.detach().cpu().numpy()[0, :])
            print(regular_mn.detach().cpu().numpy())

            label_out = label.float()
            outputs.append(label_out)
            _, _, H, W = outputs[0].shape
            all_results = torch.zeros((len(outputs), 1, H, W))
            for j in range(len(outputs)):
                if isinstance(model, RCF) or isinstance(model, HED):
                    if j == 6:
                        all_results[j, 0, :, :] = outputs[j][0, 0, :, :]
                    else:
                        all_results[j, 0, :, :] = outputs[j][0, 0, :, :] / (outputs[j][0, 0, :, :] + outputs[j][0, 1, :, :])
                else:
                    if j == 5:
                        all_results[j, 0, :, :] = outputs[j][0, 0, :, :]
                    else:
                        all_results[j, 0, :, :] = outputs[j][0, 0, :, :] / (outputs[j][0, 0, :, :] + outputs[j][0, 1, :, :])
            torchvision.utils.save_image(all_results, join(save_dir, "iter-%d.jpg" % i))
    return losses.avg, epoch_loss



def test(model, test_loader, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for idx, (image, image_gt, file_name) in enumerate(test_loader):
        image = image.cuda()
        image_gt = image_gt.cuda()
        _, _, H, W = image.shape
        results = model(image, reture_expect=True)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()
        results_all = torch.zeros((len(results) + 1, 1, H, W))
        for i in range(len(results)):
            results_all[i, 0, :, :] = results[i][0, :, :]
        results_all[i + 1, 0, :, :] = image_gt[0, :, :]
        torchvision.utils.save_image(results_all, join(save_dir, "%s.jpg" % file_name))
        result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(join(save_dir, "%s.png" % file_name))
        print("Running test [%d/%d]" % (idx + 1, len(test_loader)))

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == '__main__':
    main()