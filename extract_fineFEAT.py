import os
import sys
import argparse
import time

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

import x3d_fine

from charades_fine import Charades
from charades_fine import mt_collate_fn as collate_fn

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, MultiScaleRandomCropMultigrid, ToTensor, CenterCrop, CenterCropScaled
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

import warnings
warnings.filterwarnings("ignore")


BS = 1
BS_UPSCALE = 1
INIT_LR = 0.02 * BS_UPSCALE
X3D_VERSION = 'M'
CHARADES_MEAN = [0.413, 0.368, 0.338]
CHARADES_STD = [0.131, 0.125, 0.132]
CHARADES_TR_SIZE = 7900
CHARADES_VAL_SIZE = 1850
CHARADES_ROOT = '/data/add_disk0/kumarak/Charades_v1_rgb'
CHARADES_ANNO = 'data/charades.json'
FINE_SAVE_DIR = '/nfs/bigcornea/add_disk0/kumarak/fine_spatial7x7'
# pre-extract fine features and save here, to reduce compute req
# MAKE DIRS FINE_SAVE_DIR/['layer1', 'layer2', 'layer3', 'layer4', 'conv5']
feat_keys = ['layer1', 'layer2', 'layer3', 'layer4', 'conv5']
for k in feat_keys:
    if not os.path.exists(os.path.join(FINE_SAVE_DIR,k)):
        os.makedirs(os.path.join(FINE_SAVE_DIR,k))


# 0.00125 * BS_UPSCALE --> 80 epochs warmup 2000
def run(init_lr=INIT_LR, warmup_steps=0, max_epochs=100, root=CHARADES_ROOT,
    train_split=CHARADES_ANNO, batch_size=BS*BS_UPSCALE, frames=80, save_dir= FINE_SAVE_DIR):

    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,256.], 'XL':[360.,450.]}[X3D_VERSION] #[256.,320.]
    gamma_tau = {'S':6, 'M':5*1, 'XL':5}[X3D_VERSION] # 5

    load_steps = st_steps = steps = 0
    epochs = 0
    num_steps_per_update = 1
    cur_iterations = steps * num_steps_per_update
    iterations_per_epoch = CHARADES_TR_SIZE//(batch_size*1)
    val_iterations_per_epoch = CHARADES_VAL_SIZE//(batch_size)
    max_steps = iterations_per_epoch * max_epochs


    val_spatial_transforms = Compose([CenterCropScaled(crop_size),
                                        ToTensor(255),
                                        Normalize(CHARADES_MEAN, CHARADES_STD)])

    # SET 'TESTING' FOR BOTH, TO EXTRACT
    dataset = Charades(train_split, 'testing', root, val_spatial_transforms,
                                task='loc', frames=frames, gamma_tau=gamma_tau, crops=1, extract_feat=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=8, pin_memory=True, collate_fn=collate_fn)

    val_dataset = Charades(train_split, 'testing', root, val_spatial_transforms,
                                task='loc', frames=frames, gamma_tau=gamma_tau, crops=1, extract_feat=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=8, pin_memory=True, collate_fn=collate_fn)


    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    print('train',len(datasets['train']),'val',len(datasets['val']))
    print('Total iterations:', max_steps, 'Total epochs:', max_epochs)
    print('datasets created')

    fine_net = x3d_fine.generate_model(x3d_version=X3D_VERSION, n_classes=400, n_input_channels=3, task='loc',
                                    dropout=0.5, base_bn_splits=1, global_tower=True)

    fine_net.replace_logits(157)

    load_ckpt = torch.load('models/fine_charades_039000_SAVE.pt')
    state = fine_net.state_dict()
    state.update(load_ckpt['model_state_dict'])
    fine_net.load_state_dict(state)

    fine_net.cuda()
    fine_net = nn.DataParallel(fine_net)
    print('model loaded')

    lr = init_lr
    print ('LR:%f'%lr)


    optimizer = optim.SGD(fine_net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)
    if steps>0:
        optimizer.load_state_dict(load_ckpt['optimizer_state_dict'])
        lr_sched.load_state_dict(load_ckpt['scheduler_state_dict'])

    criterion = nn.BCEWithLogitsLoss()

    val_apm = APMeter()
    tr_apm = APMeter()

    while epochs < max_epochs:
        print ('Step {} Epoch {}'.format(steps, epochs))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']+['val']:
            bar_st = iterations_per_epoch if phase == 'train' else val_iterations_per_epoch
            bar = pkbar.Pbar(name='update: ', target=bar_st)

            fine_net.train(False)  # Set model to evaluate mode
            # FOR EVAL AGGREGATE BN STATS
            _ = fine_net.module.aggregate_sub_bn_stats()
            torch.autograd.set_grad_enabled(False)

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
            for i,data in enumerate(dataloaders[phase]):
                #for data in dataloaders[phase]:
                num_iter += 1
                bar.update(i)

                inputs, labels, masks, name = data
                b,n,c,t,h,w = inputs.shape
                inputs = inputs.view(b*n,c,t,h,w)

                inputs = inputs.cuda() # B 3 T W H
                tl = labels.size(2)
                labels = labels.cuda() # B C TL
                masks = masks.cuda() # B TL
                valid_t = torch.sum(masks, dim=1).int()

                feat,_ = fine_net([inputs, masks]) # N C T 1 1
                keys = list(feat.keys())
                print(i, name[0], feat[keys[0]].cpu().numpy().shape, feat[keys[1]].cpu().numpy().shape,
                        feat[keys[2]].cpu().numpy().shape, feat[keys[3]].cpu().numpy().shape, feat[keys[4]].cpu().numpy().shape)
                for k in feat:
                    torch.save(feat[k].data.cpu(), os.path.join(save_dir, k, name[0]))
        break


if __name__ == '__main__':
    run()
