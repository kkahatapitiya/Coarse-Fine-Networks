import os
import sys
import argparse
import time
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0', type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
#import videotransforms
from torchsummary import summary

import numpy as np
import pkbar
from apmeter import APMeter

import x3d_coarse

from charades_coarse_fineFEAT import Charades
from charades_coarse_fineFEAT import mt_collate_fn as collate_fn

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, MultiScaleRandomCropMultigrid, ToTensor, CenterCrop, CenterCropScaled, CornerCrop
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel


import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)

BS = 1 #5 #15 #15 #15 #15 #15 #6 #6 #16
BS_UPSCALE = 1 #4
INIT_LR = 0.02 * BS_UPSCALE #* 0.1
X3D_VERSION = 'M'
CHARADES_MEAN = [0.413, 0.368, 0.338]
CHARADES_STD = [0.131, 0.125, 0.132] # CALCULATED ON CHARADES TRAINING SET FOR FRAME-WISE MEANS
CHARADES_TR_SIZE = 7900
CHARADES_VAL_SIZE = 1850
CHARADES_ROOT = '/home/kkahatapitiy/data/Charades_v1_rgb'
CHARADES_ANNO = 'data/charades.json'
FINE_FEAT_DIR = '/home/kkahatapitiy/data/x3d_feat/originalDGX' # originalDGX original bg cm mu # pre-extract fine features and save here, to reduce compute req


# 0.00125 * BS_UPSCALE --> 80 epochs warmup 2000
def run(init_lr=INIT_LR, warmup_steps=0, max_epochs=200, root=CHARADES_ROOT, train_split=CHARADES_ANNO,
    batch_size=BS*BS_UPSCALE, frames=80*4, fine_feat = FINE_FEAT_DIR):

    feat_keys = ['layer1', 'layer2', 'layer3', 'layer4', 'conv5'] #, 'gx']
    feat_depth = {'layer1':24, 'layer2':48, 'layer3':96, 'layer4':192, 'conv5':432}

    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,320.], 'XL':[360.,450.]}[X3D_VERSION] #[256.,320.]
    gamma_tau = {'S':6, 'M':5*1, 'XL':5}[X3D_VERSION] # 5

    load_steps = st_steps = steps = 0
    epochs = 0
    num_steps_per_update = 1 # accum gradient
    cur_iterations = steps * num_steps_per_update
    iterations_per_epoch = CHARADES_TR_SIZE//batch_size
    val_batch_size = batch_size#//5*2 # //8 for 10 crop with 4 gpus, //2 otherwise
    val_iterations_per_epoch = CHARADES_VAL_SIZE//val_batch_size
    max_steps = iterations_per_epoch * max_epochs


    train_spatial_transforms = Compose([MultiScaleRandomCropMultigrid([crop_size/i for i in resize_size], crop_size),
                                        RandomHorizontalFlip(),
                                        ToTensor(255),
                                        Normalize(CHARADES_MEAN, CHARADES_STD)])
    val_spatial_transforms = Compose([ CenterCropScaled(crop_size),
                                        ToTensor(255),
                                        Normalize(CHARADES_MEAN, CHARADES_STD)])

    dataset = Charades(train_split, 'training', root, fine_feat, feat_keys, train_spatial_transforms,
                                task='loc', frames=frames, gamma_tau=gamma_tau, crops=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=16, pin_memory=True, collate_fn=collate_fn)

    val_dataset = Charades(train_split, 'testing', root, fine_feat, feat_keys, val_spatial_transforms,
                                task='loc', frames=frames, gamma_tau=gamma_tau, crops=1)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
                                num_workers=16, pin_memory=True, collate_fn=collate_fn)


    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    print('train',len(datasets['train']),'val',len(datasets['val']))
    print('Total iterations:', max_steps, 'Total epochs:', max_epochs)
    print('datasets created')

    # setup the model
    # ON 4 GPUS, 128/4, 32 CLIPS PER GPU, base_bn_splits=4 means BN calculated per 8xlong_cycle_multiplier clips

    coarse_net_cm = x3d_coarse.generate_model(x3d_version=X3D_VERSION, n_classes=400+0, n_input_channels=3,
                                    feat_depth=feat_depth, task='loc', dropout=0.5, base_bn_splits=1,
                                    learnedMixing=True, isMixing=True, t_pool='grid') #'grid'
    coarse_net_mu = x3d_coarse.generate_model(x3d_version=X3D_VERSION, n_classes=400+0, n_input_channels=3,
                                    feat_depth=feat_depth, task='loc', dropout=0.5, base_bn_splits=1,
                                    learnedMixing=True, isMixing=True, t_pool='grid') #'grid'
    coarse_net_bg = x3d_coarse.generate_model(x3d_version=X3D_VERSION, n_classes=400+0, n_input_channels=3,
                                    feat_depth=feat_depth, task='loc', dropout=0.5, base_bn_splits=1,
                                    learnedMixing=True, isMixing=True, t_pool='grid') #'grid'
    '''#load_ckpt = torch.load('models/x3d_multigrid_kinetics_fb_pretrained.pt')
    #load_ckpt = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_weakly_fromFB_full_617000.pt')
    #load_ckpt = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_randReplace_MixUp_x1Adaptive_CrEntropyObj_fromFB_full_610000.pt') # 560000
    load_ckpt = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_randReplace_CutMixV2_3_x1Adaptive_CrEntropyObj_fromFB_full_600000.pt')
    #coarse_net.load_state_dict(load_ckpt['model_state_dict'])
    state = coarse_net.state_dict()
    state.update(load_ckpt['model_state_dict'])
    coarse_net.load_state_dict(state)'''

    '''load_ckpt = torch.load('models/x3d_multigrid_kinetics_fb_pretrained.pt')
    state_to_load = load_ckpt['model_state_dict']
    state = coarse_net.state_dict()
    for k in state_to_load:
        k_ = k
        if 'split_bn' in k: continue
        elif 'bn.' in k: k_ = k.replace('bn.','')
        if state_to_load[k].shape != state[k_].shape:
            if 'running_mean' in k or 'running_var' in  k:
                n = state_to_load[k].shape[0]
                state[k_] = F.adaptive_avg_pool1d(state_to_load[k].view(1,1,n), n//2*4).view(-1)
            elif 'fc' in k:
                state[k_] = state_to_load[k].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                print(k, state_to_load[k].shape, state[k_].shape)
        else:
            state[k_] = state_to_load[k]
    coarse_net.load_state_dict(state)'''


    #save_model = 'models/coarse_fineFEAT_charades_cm_' cm+originalFeat
    #save_model = 'models/coarse_fineFEAT_charades_mu+originalFeat_'
    #save_model = 'models/coarse_fineFEAT_charades_bg+originalFeat_'
    #save_model = 'models/coarse_fineFEAT_charades_original+cmFeat_'
    #save_model = 'models/coarse_fineFEAT_charades_cm+cm2Feat_'
    save_model = 'models/temp_'

    coarse_net_mu.replace_logits(157)
    coarse_net_cm.replace_logits(157)
    coarse_net_bg.replace_logits(157)

    #load_ckpt = torch.load('models/coarse_fineFEAT_charades_019000_SAVE.pt')

    #load_ckpt = torch.load('models/x3d_grid_charades_loc_rgb_sgd_bg_049000.pt')
    #load_ckpt = torch.load('models/x3d_grid_charades_loc_rgb_sgd_cm_049000.pt')
    #load_ckpt = torch.load('models/x3d_grid_charades_loc_rgb_sgd_mu_049000.pt')            coarse          fine

    #load_ckpt = torch.load('models/x3d_charades_loc_GRIDPOOL4_rgb_sgd_056000_SAVE.pt')     # mAP: 0.1796   mAP: 0.1774

    #load_ckpt = torch.load('models/x3d_grid_charades_loc_rgb_sgd_bg_049000_DGX.pt')         # mAP: 0.1778   mAP: 0.1879
    #load_ckpt = torch.load('models/x3d_grid_charades_loc_rgb_sgd_cm_047000_DGX.pt')        # mAP: 0.1885   mAP: 0.1899  -->  0.2283
    #load_ckpt = torch.load('models/x3d_grid_charades_loc_rgb_sgd_mu_048000_DGX.pt')        # mAP: 0.1897   mAP: 0.1918  -->  0.2312

    #load_ckpt = torch.load('models/x3d_grid_charades_loc_rgb_sgd_original_049000_DGX.pt')  # mAP: 0.1813   mAP: 0.1728  -->  0.2329

    #cm + original_feat     mAP: 0.2357
    #mu + original_feat     mAP: 0.2362
    #bf + original_feat     mAP: 0.2365

    #original + cm_feat     mAP: 0.2321
    #cm + mu_feat           mAP: 0.2311

    #load_ckpt = torch.load('models/x3d_charades_loc_rgb_sgd_mu_049000_DGX.pt')

    load_ckpt_cm = torch.load('models/coarse_fineFEAT_charades_cm+originalFeat_016000.pt')
    load_ckpt_mu = torch.load('models/coarse_fineFEAT_charades_mu+originalFeat_021000.pt')
    load_ckpt_bg = torch.load('models/coarse_fineFEAT_charades_bg+originalFeat_018000.pt')


    coarse_net_mu.load_state_dict(load_ckpt_mu['model_state_dict'])
    coarse_net_bg.load_state_dict(load_ckpt_bg['model_state_dict'])
    coarse_net_cm.load_state_dict(load_ckpt_cm['model_state_dict'])

    if steps>0:
        load_ckpt = torch.load('models/coarse_fineFEAT_charades_original_'+str(load_steps).zfill(6)+'.pt')
        coarse_net_cm.load_state_dict(load_ckpt['model_state_dict'])

    coarse_net_cm.cuda()
    coarse_net_mu.cuda()
    coarse_net_bg.cuda()
    coarse_net_cm = nn.DataParallel(coarse_net_cm)
    coarse_net_mu = nn.DataParallel(coarse_net_mu)
    coarse_net_bg = nn.DataParallel(coarse_net_bg)
    print('model loaded')

    lr = init_lr
    print ('LR:%f'%lr)


    rw_params=[]; base_params=[];
    for name,para in coarse_net_cm.named_parameters():
        if 'rw' in name or 'mix' in name: rw_params.append(para)
        else: base_params.append(para)
    optimizer = optim.SGD([{'params': base_params}, {'params': rw_params, 'lr': lr*10}], lr=lr, momentum=0.9, weight_decay=1e-5) # lr*10

    #optimizer = optim.SGD(coarse_net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

    lr_schedule = [15,25,35]
    #lr_schedule = [20,30,40]
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule, verbose=True)
    #lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1, verbose=True) #2
    if steps>0:
        optimizer.load_state_dict(load_ckpt['optimizer_state_dict'])
        lr_sched.load_state_dict(load_ckpt['scheduler_state_dict'])

    criterion_class = nn.BCELoss(reduction='mean')
    criterion_loc = nn.BCELoss(reduction='sum')

    val_apm = APMeter()
    tr_apm = APMeter()
    write_file = open('localize_corr_v1_weakSupEnsemble4.csv', 'w', newline='\n')
    writer = csv.writer(write_file)

    while epochs < max_epochs:
        print ('Step {} Epoch {}'.format(steps, epochs))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['val']: #2*['train']+['val']: #['val']:# for training --> 2*['train']+['val']:
            bar_st = iterations_per_epoch if phase == 'train' else val_iterations_per_epoch
            bar = pkbar.Pbar(name='update: ', target=bar_st)
            if phase == 'train':
                coarse_net_cm.train(True)
                epochs += 1
                torch.autograd.set_grad_enabled(True)
                torch.cuda.empty_cache()
            else:
                coarse_net_cm.train(False)  # Set model to evaluate mode
                coarse_net_bg.train(False)
                coarse_net_mu.train(False)
                # FOR EVAL AGGREGATE BN STATS
                _ = coarse_net_cm.module.aggregate_sub_bn_stats()
                _ = coarse_net_mu.module.aggregate_sub_bn_stats()
                _ = coarse_net_bg.module.aggregate_sub_bn_stats()
                torch.autograd.set_grad_enabled(False)
                torch.cuda.empty_cache()

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            tot_dis_loss = 0.0
            tot_acc = 0.0
            tot_corr = 0.0
            tot_dat = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data.
            print(phase)
            lines=[]
            for i,data in enumerate(dataloaders[phase]):
                num_iter += 1
                bar.update(i)
                if phase == 'train':
                    inputs, labels, masks, feat, feat_masks, meta, name, dur = data
                    inputs = inputs.squeeze(1)
                else:
                    inputs, labels, masks, feat, feat_masks, meta, name, dur = data
                    b,n,c,t,h,w = inputs.shape
                    inputs = inputs.view(b*n,c,t,h,w)
                    if n>1 and b != val_batch_size:
                        continue


                inputs = inputs.cuda() # B 3 T W H
                tl = labels.size(2)
                labels = labels.cuda() # B C TL
                masks = masks.cuda() # B TL
                for k in feat_keys:
                    feat[k] = feat[k].cuda()
                feat_masks = feat_masks.cuda()
                meta = meta.cuda()
                valid_t = torch.sum(masks, dim=1).int()


                per_frame_logits = 0
                for net in [coarse_net_mu, coarse_net_bg, coarse_net_cm]:
                    t_lim_inference = 2048 #512 #1000 # if VAL data too large to fit into memory, split
                    if phase == 'train' or inputs.shape[2]<t_lim_inference+5:
                        per_frame_logits_single = net([inputs, feat, feat_masks, i, meta]) # B C T
                    else:
                        temp_logit=[]
                        for t_ind in range(0,inputs.shape[2]//t_lim_inference+1):
                            input_t_ind = inputs[:,:,t_ind*t_lim_inference:min(inputs.shape[2],(t_ind+1)*t_lim_inference),:,:]
                            temp_logit.append(net([input_t_ind, feat, feat_masks, i, meta])) # B C T
                            meta[:,0] += t_lim_inference
                        per_frame_logits_single = torch.cat(temp_logit, dim=2)

                    #per_frame_logits_single = F.interpolate(per_frame_logits_single, tl, mode='linear')
                    per_frame_logits += per_frame_logits_single
                per_frame_logits = per_frame_logits/3
                per_frame_logits = F.interpolate(per_frame_logits, tl, mode='linear')

                if phase == 'train':
                    probs = F.sigmoid(per_frame_logits) * masks.unsqueeze(1)
                else:
                    per_frame_logits = per_frame_logits.view(b,n,-1,tl)
                    probs = F.sigmoid(per_frame_logits) #* masks.unsqueeze(1)
                    probs = torch.max(probs, dim=1)[0] * masks.unsqueeze(1)
                    per_frame_logits = torch.max(per_frame_logits, dim=1)[0]

                cls_loss = criterion_class(torch.max(probs, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.item()

                loc_loss = criterion_loc(probs, labels) / (torch.sum(masks)*labels.shape[1])
                tot_loc_loss += loc_loss.item() #data[0]

                if phase == 'train':
                    for b in range(labels.shape[0]):
                        tr_apm.add(probs[b][:,:valid_t[b].item()].transpose(0,1).detach().cpu().numpy(),
                                    labels[b][:,:valid_t[b].item()].transpose(0,1).cpu().numpy())
                else:
                    for b in range(labels.shape[0]):

                        val_apm.add(probs[b][:,:valid_t[b].item()].transpose(0,1).detach().cpu().numpy(),
                                    labels[b][:,:valid_t[b].item()].transpose(0,1).cpu().numpy())

                        p1 = probs[b][:,:valid_t[b].item()]
                        l1 = labels[b][:,:valid_t[b].item()]
                        sc = valid_t[b].item()/25. #p1.shape[1]/25.
                        du = dur[b].item()/25.
                        p1 = p1[:,1::int(sc)][:,:25]
                        l1 = l1[:,1::int(sc)][:,:25] #

                        a = p1.transpose(0,1).detach().cpu().numpy() # T C

                        for i in range(a.shape[0]):
                            st=''
                            act = list(a[i])
                            for j in range(len(act)):
                                st+=str(act[j])+' '
                            st=st[:-1]
                            writer.writerow([name[0], 1+i*du, st])

                        '''val_apm.add(p1.transpose(0,1).detach().cpu().numpy(),
                                    l1.transpose(0,1).cpu().numpy())'''



                loss = 1 * (cls_loss + loc_loss)/(2 * num_steps_per_update)
                tot_loss += loss.item() #data[0]

                if phase == 'train':
                    loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    #lr_warmup(lr, steps-st_steps, warmup_steps, optimizer) # steps init_lr, USE ONLY AT THE START
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    s_times = iterations_per_epoch//2#100*4
                    if steps % s_times == 0:
                        tr_map = tr_apm.value().mean()
                        tr_apm.reset()
                        print (' Epoch:{} {} steps: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}'.format(epochs, phase,
                            steps, tot_loc_loss/(s_times*num_steps_per_update), tot_cls_loss/(s_times*num_steps_per_update), tot_loss/s_times, tr_map))#, tot_acc/(s_times*num_steps_per_update)))
                        tot_loss = tot_loc_loss = tot_cls_loss = tot_dis_loss = tot_acc = tot_corr = tot_dat = 0.
                    if steps % (1000) == 0:
                        ckpt = {'model_state_dict': coarse_net.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': lr_sched.state_dict()}
                        torch.save(ckpt, save_model+str(steps).zfill(6)+'.pt')
            if phase == 'val':
                write_file.close()
                val_map = val_apm.value().mean()
                val_apm.reset()
                print (' Epoch:{} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}'.format(epochs, phase,
                    tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, val_map))#, tot_acc/num_iter))
                tot_loss = tot_loc_loss = tot_cls_loss = tot_dis_loss = tot_acc = tot_corr = tot_dat = 0.
                lr_sched.step()
                #lr_sched.step(tot_loss)


def lr_warmup(init_lr, cur_steps, warmup_steps, opt):
    start_after = 1
    if cur_steps < warmup_steps and cur_steps > start_after:
        lr_scale = min(1., float(cur_steps + 1) / warmup_steps)
        for pg in opt.param_groups:
            pg['lr'] = lr_scale * init_lr


if __name__ == '__main__':
    run()
