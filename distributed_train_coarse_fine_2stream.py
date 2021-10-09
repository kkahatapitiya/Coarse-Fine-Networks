import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
import time
from collections import Counter
import builtins
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.distributed as dist

import torchvision
from torchvision import datasets, transforms
#import videotransforms
from torchsummary import summary

import numpy as np
from barbar import Bar
import pkbar
from apmeter import APMeter

#import x3d_classi3 as resnet_x3d
#import x3d_fb_withFeat_2stream as resnet_x3d

#import x3d_fb_withFeat as resnet_x3d
import x3d_coarse
import x3d_fine

#from kinetics_multigrid import Kinetics
#from kinetics import Kinetics as Kinetics_val
from charades_coarse_fine_2stream import Charades
from charades_coarse_fine_2stream import mt_collate_fn as collate_fn

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, MultiScaleRandomCropMultigrid, ToTensor, CenterCrop, CenterCropScaled
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
#parser.add_argument('-gpu', default='0', type=str)
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

args = parser.parse_args()


BS = 4 #3 #5 #15 #6 #6 #16
BS_UPSCALE = 1 #4
GPUS = 4 #3 #3
INIT_LR = 0.02 * BS_UPSCALE #* 0.1
X3D_VERSION = 'M'
CHARADES_MEAN = [0.413, 0.368, 0.338]
CHARADES_STD = [0.131, 0.125, 0.132] # CALCULATED ON CHARADES TRAINING SET FOR FRAME-WISE MEANS
CHARADES_TR_SIZE = 7900
CHARADES_VAL_SIZE = 1850
CHARADES_ROOT = '/home/kkahatapitiy/data/Charades_v1_rgb'
CHARADES_ANNO = 'data/charades.json'
FINE_FEAT_DIR = '/home/kkahatapitiy/data/x3d_feat/cm' # original bg cm mu # pre-extract fine features and save here, to reduce compute req


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.local_rank != -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        assert args.local_rank == -1
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(run, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        run(args.gpu, ngpus_per_node, args)


# 0.00125 * BS_UPSCALE --> 80 epochs warmup 2000
def run(gpu, ngpus_per_node, args):

    init_lr=INIT_LR
    warmup_steps=0
    max_epochs=200
    mode='rgb'
    root=CHARADES_ROOT #'/nfs/bigdisk/kumarak/datasets/charades/Charades_v1_rgb'
    train_split=CHARADES_ANNO
    batch_size=BS*BS_UPSCALE
    frames=80
    x3d_out = ['','']

    feat_keys = ['layer1', 'layer2', 'layer3', 'layer4', 'conv5'] #['layer1', 'layer2', 'layer3', 'layer4', 'conv5']
    feat_depth = {'layer1':24, 'layer2':48, 'layer3':96, 'layer4':192, 'conv5':432}

    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,320.], 'XL':[360.,450.]}[X3D_VERSION] #[256.,320.]
    gamma_tau = {'S':6, 'M':5*1, 'XL':5}[X3D_VERSION] # 5

    # *2 FOR REDUCING BS FROM 32 TO 16
    steps = load_steps = st_steps = 0 #24000 #9000 #0
    epochs = 0 #30 #18 #0
    num_steps_per_update = 1 #4 * 2 # accum gradient
    cur_iterations = steps * num_steps_per_update
    iterations_per_epoch = 7900//(batch_size * GPUS) # *num_steps_per_update
    val_batch_size = batch_size#//5*2 #5 #batch_size//3 #*2 #//2 # //8 for 10 crop with 4 gpus, //2 otherwise
    val_iterations_per_epoch = 1850//(val_batch_size * GPUS)
    max_steps = iterations_per_epoch * max_epochs


    args.gpu = gpu

    if args.distributed:
        if args.local_rank != -1:
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    args.print = args.gpu == 0
    # suppress printing if not master
    if (args.multiprocessing_distributed and args.gpu != 0) or\
       (args.local_rank != -1 and args.gpu != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass


    print("=> creating X3D model with backbone")
    x3d = x3d_coarse.generate_model(x3d_version=X3D_VERSION, n_classes=400+0, n_input_channels=3,
                                    feat_depth=feat_depth, task='loc', dropout=0.5, base_bn_splits=1,
                                    learnedMixing=True, isMixing=True, t_pool='grid') #'grid'
    #x3d_g = resnet_x3d_global.generate_model(x3d_version=X3D_VERSION, n_classes=400, n_input_channels=3,
    #                                task='loc', dropout=0.5, base_bn_splits=1, global_tower=True)
    x3d_g = x3d_fine.generate_model(x3d_version=X3D_VERSION, n_classes=400, n_input_channels=3, task='loc',
                                    dropout=0., base_bn_splits=1, global_tower=True)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            x3d.cuda(args.gpu)
            x3d_g.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            x3d = torch.nn.parallel.DistributedDataParallel(x3d, device_ids=[args.gpu], find_unused_parameters=True)
            x3d_g = torch.nn.parallel.DistributedDataParallel(x3d_g, device_ids=[args.gpu], find_unused_parameters=True)
            x3d_without_ddp = x3d.module
            x3d_g_without_ddp = x3d_g.module
        else:
            x3d.cuda()
            x3d_g.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            x3d = torch.nn.parallel.DistributedDataParallel(x3d, find_unused_parameters=True)
            x3d_g = torch.nn.parallel.DistributedDataParallel(x3d_g, find_unused_parameters=True)
            x3d_without_ddp = x3d.module
            x3d_g_without_ddp = x3d_g.module
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        x3d = x3d.cuda(args.gpu)
        x3d_g = x3d_g.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")


    train_spatial_transforms = Compose([MultiScaleRandomCropMultigrid([crop_size/i for i in resize_size], crop_size),
                                        RandomHorizontalFlip(),
                                        ToTensor(255),
                                        Normalize(CHARADES_MEAN, CHARADES_STD)])
    val_spatial_transforms = Compose([CenterCropScaled(crop_size),
                                        ToTensor(255),
                                        Normalize(CHARADES_MEAN, CHARADES_STD)])

    dataset = Charades(train_split, 'training', root, x3d_out, feat_keys, mode, train_spatial_transforms,
                                task='loc', frames=frames, gamma_tau=gamma_tau, crops=1)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None),
                                num_workers=8, pin_memory=True, sampler=sampler, collate_fn=collate_fn)

    val_dataset = Charades(train_split, 'testing', root, x3d_out, feat_keys, mode, val_spatial_transforms,
                                task='loc', frames=frames, gamma_tau=gamma_tau, crops=1)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=(val_sampler is None),
                                num_workers=8, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)


    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    print('train',len(datasets['train']),'val',len(datasets['val']))
    print('Total iterations:', max_steps, 'Total epochs:', max_epochs)
    print('datasets created')



    '''load_ckpt = torch.load('models/x3d_multigrid_kinetics_fb_pretrained.pt')
    #x3d.load_state_dict(load_ckpt['model_state_dict'])
    state = x3d.state_dict()
    state.update(load_ckpt['model_state_dict'])
    x3d.load_state_dict(state)
    state_g = x3d_g.state_dict()
    state_g.update(load_ckpt['model_state_dict'])
    x3d_g.load_state_dict(state_g)'''

    #save_model = 'models/x3d_charades_loc_rgb_withFeat_2stream_sgd_'
    #save_model = 'models/x3d_charades_loc_rgb_withFeat_2stream_V2_sgd_'
    save_model = 'models/x3d_coarse_fine_charades_loc_rgb_2stream_sgd_'

    #x3d.replace_logits(157)
    #x3d_g.replace_logits(157)

    #models/x3d_charades_loc_rgb_sgd_RANDOMSCALE_024000.pt      16.70 both
    #models/x3d_charades_loc_POOL_rgb_sgd_117000_SAVE.pt (128)  18.14
    #models/x3d_charades_loc_GRIDPOOL2_rgb_sgd_026000_SAVE.pt   17.85
    #models/x3d_charades_loc_POOL_rgb_sgd_077000_SAVE.pt        16.61
    #models/x3d_charades_loc_GRIDPOOL4_rgb_sgd_056000_SAVE.pt   18.11

    #models/x3d_charades_loc_gamma5_rgb_sgd_059000_SAVE.pt      16.74
    #models/x3d_charades_loc_rgb_sgd_039000_SAVE.pt             17.15
    #models/x3d_charades_loc_rgb_withFeatNEW_sgd_018000_SAVE.pt 20.04

    #load_ckpt = torch.load('models/x3d_grid_charades_loc_rgb_sgd_bg_049000_DGX.pt')        # mAP: 0.1778   mAP: 0.1879
    #load_ckpt = torch.load('models/x3d_grid_charades_loc_rgb_sgd_cm_047000_DGX.pt')        # mAP: 0.1885   mAP: 0.1899
    #load_ckpt = torch.load('models/x3d_grid_charades_loc_rgb_sgd_mu_048000_DGX.pt')        # mAP: 0.1897   mAP: 0.1918
    #load_ckpt = torch.load('models/x3d_grid_charades_loc_rgb_sgd_original_049000_DGX.pt')  # mAP: 0.1813   mAP: 0.1728

    #load_ckpt = torch.load('models/git/fine_charades_039000_SAVE.pt')                      # mAP: 0.1774
    #load_ckpt = torch.load('models/x3d_charades_loc_rgb_sgd_bg_049000_DGX.pt')             # mAP: 0.1879
    #load_ckpt = torch.load('models/x3d_charades_loc_rgb_sgd_cm_047000_DGX.pt')             # mAP: 0.1899
    #load_ckpt = torch.load('models/x3d_charades_loc_rgb_sgd_mu_049000_DGX.pt')             # mAP: 0.1918
    #load_ckpt = torch.load('models/x3d_charades_loc_rgb_sgd_original_049000_DGX.pt')       # mAP: 0.1728

    #load_ckpt = torch.load('models/x3d_multigrid_kinetics_fb_pretrained.pt')
    #load_ckpt = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_weakly_fromFB_full_617000.pt')
    #load_ckpt = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_randReplace_MixUp_x1Adaptive_CrEntropyObj_fromFB_full_610000.pt') # 560000
    load_ckpt = torch.load('models/x3d_multigrid_kinetics_rgb_sgd_randReplace_CutMixV2_3_x1Adaptive_CrEntropyObj_fromFB_full_600000.pt')

    '''load_ckpt = torch.load('models/x3d_grid_charades_loc_rgb_sgd_cm_047000_DGX.pt')
    state = x3d.state_dict()
    state.update(load_ckpt['model_state_dict'])
    x3d.load_state_dict(state)

    load_ckpt_g = torch.load('models/x3d_charades_loc_rgb_sgd_cm_047000_DGX.pt')
    state_g = x3d_g.state_dict()
    state_g.update(load_ckpt_g['model_state_dict'])
    x3d_g.load_state_dict(state_g)'''


    state_to_load = load_ckpt['model_state_dict']
    state = x3d_without_ddp.state_dict()
    state_g = x3d_g_without_ddp.state_dict()
    for k in state_to_load:
        k_ = k
        if 'split_bn' in k: continue
        elif 'bn.' in k: k_ = k.replace('bn.','')
        state[k_] = state_to_load[k]
        state_g[k_] = state_to_load[k]
    #state.update(load_ckpt['model_state_dict'])
    x3d_without_ddp.load_state_dict(state)
    x3d_g_without_ddp.load_state_dict(state_g)

    x3d_without_ddp.replace_logits(157)
    x3d_g_without_ddp.replace_logits(157)


    load_ckpt = torch.load('models/x3d_coarse_fine_charades_loc_rgb_2stream_sgd_032000.pt')

    state = x3d_without_ddp.state_dict()
    state.update(load_ckpt['model_l_state_dict'])
    x3d_without_ddp.load_state_dict(state)

    state_g = x3d_g_without_ddp.state_dict()
    state_g.update(load_ckpt['model_g_state_dict'])
    x3d_g_without_ddp.load_state_dict(state_g)

    if steps>0:
        load_ckpt = torch.load('models/x3d_charades_loc_rgb_withFeat_f1_2stream_v4_NEW_sgd_'+str(load_steps).zfill(6)+'.pt')
        x3d_without_ddp.load_state_dict(load_ckpt['model_l_state_dict'])
        x3d_g_without_ddp.load_state_dict(load_ckpt['model_g_state_dict'])
        #state = x3d.state_dict()
        #state.update(load_ckpt['model_state_dict'])
        #x3d.load_state_dict(state)

    #x3d.cuda()
    #x3d_g.cuda()


    '''x3d.freeze()
    x3d_g.freeze()
    for name, param in x3d.named_parameters():
        if param.requires_grad:print('updating: {}'.format(name)) #print(name)
    print('*****')
    for name, param in x3d_g.named_parameters():
        if param.requires_grad:print('updating: {}'.format(name)) #print(name)'''


    #x3d = nn.DataParallel(x3d)
    #x3d_g = nn.DataParallel(x3d_g)
    print('model loaded')

    lr = init_lr #* batch_size/len(datasets['train'])
    print ('LR:%f'%lr)


    rw_params=[]; base_params=[];
    for name,para in x3d.named_parameters():
        if 'rw' in name or 'mix' in name: rw_params.append(para)
        else: base_params.append(para)
    for name,para in x3d_g.named_parameters():
        #rw_params.append(para)
        base_params.append(para)
        #if 'pool_1' in name: rw_params.append(para)
        #else: base_params.append(para)

    optimizer = optim.SGD([{'params': base_params}, {'params': rw_params, 'lr': lr * 1}], lr=lr, momentum=0.9, weight_decay=1e-5)

    #optimizer = optim.SGD(list(x3d.parameters())+list(x3d_g.parameters()), lr=lr, momentum=0.9, weight_decay=1e-5)
    #optimizer = optim.Adam(list(x3d.parameters())+list(x3d_g.parameters()), lr=0.001, weight_decay=1e-5)
    #lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    #lr_schedule = [20,30,40]
    #lr_schedule = [15,25,35]
    #lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule, verbose=True)
    #lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1, verbose=True) #2
    #for _ in range(steps):
    #    lr_sched.step()

    lr_schedule = [30,45,55] #[15,25,35]
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule, verbose=True)

    if steps>0:
        optimizer.load_state_dict(load_ckpt['optimizer_state_dict'])
        lr_sched.load_state_dict(load_ckpt['scheduler_state_dict'])

        '''state = load_ckpt['optimizer_state_dict']
        state['param_groups'][0]['lr'] = 0.001
        state['param_groups'][1]['lr'] = 0.01
        optimizer.load_state_dict(state)
        #print(optimizer.param_groups[0]['lr'])

        state = load_ckpt['scheduler_state_dict']
        state['milestones']=Counter([25,35])
        state['_last_lr']=[0.001, 0.01]
        #state['num_bad_epochs']=0
        #state['patience']=2
        lr_sched.load_state_dict(state)'''

        #state = load_ckpt['scheduler_state_dict']
        #state['num_bad_epochs']=0
        #state['patience']=2
        #lr_sched.load_state_dict(state)
    #lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, min_lr=1e-3, verbose=True)

    #criterion = nn.BCEWithLogitsLoss()
    criterion_class = nn.BCELoss(reduction='mean').cuda(args.gpu)
    criterion_loc = nn.BCELoss(reduction='sum').cuda(args.gpu)
    #for g in optimizer.param_groups:
        #if '_rw' in name:
        #print(g)
        #g['lr'] *= long_cycle_lr_scale[long_ind]

    #num_steps_per_update = 4 * 2 # accum gradient
    #steps = 0#1200#1000 #1800 #400 #0
    #epochs = 0#30#25 #44 #10 #0
    val_apm = APMeter()
    tr_apm = APMeter()
    # train it
    while epochs < max_epochs:#for epoch in range(num_epochs):
        print ('Step {} Epoch {}'.format(steps, epochs))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['val']: #2*['train']+['val']: #*5+['val']:
            bar_st = iterations_per_epoch if phase == 'train' else val_iterations_per_epoch
            bar = pkbar.Pbar(name='update: ', target=bar_st)
            if phase == 'train':
                x3d.train(True)
                x3d_g.train(True)
                #_ = x3d_g.module.aggregate_sub_bn_stats()
                #i3d2.train(True)
                epochs += 1
                torch.autograd.set_grad_enabled(True)
            else:
                x3d.train(False)  # Set model to evaluate mode
                x3d_g.train(False)
                # FOR EVAL AGGREGATE BN STATS
                #_ = x3d.module.aggregate_sub_bn_stats()
                #_ = x3d_g.module.aggregate_sub_bn_stats()
                #i3d2.train(False)
                torch.autograd.set_grad_enabled(False)

            if args.distributed:
                dataloader.sampler.set_epoch(epochs)
                val_dataloader.sampler.set_epoch(epochs)

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            tot_loc_loss_g = 0.0
            tot_cls_loss_g = 0.0
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
                if phase == 'train':
                    #inputs, labels, masks, inputs_g, labels_g, masks_gl, feat, feat_masks, meta = data
                    inputs, labels, masks, inputs_g, labels_g, masks_gl, meta, name, dur = data
                    inputs = inputs.squeeze(1)
                    inputs_g = inputs_g.squeeze(1)
                else:
                    #if i> 10:#iterations_per_epoch:
                    #    break
                    #inputs, labels, masks, inputs_g, labels_g, masks_gl, feat, feat_masks, meta = data
                    inputs, labels, masks, inputs_g, labels_g, masks_gl, meta, name, dur = data
                    b,n,c,t,h,w = inputs.shape
                    inputs = inputs.view(b*n,c,t,h,w)
                    bg,ng,cg,tg,hg,wg = inputs_g.shape
                    inputs_g = inputs_g.view(bg*ng,cg,tg,hg,wg)
                    #if b != val_batch_size:
                    #    continue

                # Gamma_tau sampling ****** MAY NOT BE GOOD FOR LOCALIZATION TASKS ******
                #inputs = inputs[:,:,::gamma_tau,:,:]
                #labels = labels[:,:,::gamma_tau]

                # wrap them in Variable
                inputs = inputs.cuda() # B 3 T W H
                #labels = torch.max(labels, dim=2)[0] # B C T --> B C
                labels = labels.cuda() # B C TL
                masks = masks.cuda() # B TL

                inputs_g = inputs_g.cuda()
                labels_g = labels_g.cuda()
                masks_gl = masks_gl.cuda() # g label mask
                masks_g = masks_gl[:,::gamma_tau*2].cuda() # g clip mask

                '''for k in feat_keys:
                    feat[k] = feat[k].cuda()
                feat_masks = feat_masks.cuda()'''

                tl = labels.size(2)
                tl_g = labels_g.size(2)
                valid_t = torch.sum(masks, dim=1).int()
                meta = meta.cuda()
                #labels = labels.unsqueeze(1).cuda() #[:,:50,:] # B 1


                #print(inputs_g.device, masks_g.device)
                feat_g, per_frame_logits_g = x3d_g([inputs_g, masks_g]) # B C T
                #per_frame_logits_g, feat_g = x3d_g(inputs_g) # B C T
                #logits_g = logits_g.squeeze(2) # B C
                #feat_g = x3d_g(inputs_g) # B C T

                per_frame_logits = x3d([inputs, feat_g, masks_g, i, meta]) # B C T

                '''t_lim_inference = 1024 #512 #1000 # if VAL data too large to fit into memory, split
                if phase == 'train' or inputs.shape[2]<t_lim_inference+5:
                    per_frame_logits = x3d([inputs, feat_g, masks_g, i, meta]) # B C T
                else:
                    temp_logit=[]
                    for t_ind in range(0,inputs.shape[2]//t_lim_inference+1):
                        input_t_ind = inputs[:,:,t_ind*t_lim_inference:min(inputs.shape[2],(t_ind+1)*t_lim_inference),:,:]
                        temp_logit.append(x3d([input_t_ind, feat_g, masks_g, i, meta])) # B C T
                        meta[:,0] += t_lim_inference
                    per_frame_logits = torch.cat(temp_logit, dim=2)'''

                per_frame_logits = F.interpolate(per_frame_logits, tl, mode='linear')
                per_frame_logits_g = F.interpolate(per_frame_logits_g, tl_g, mode='linear')

                if phase == 'train':
                    probs = F.sigmoid(per_frame_logits) * masks.unsqueeze(1)
                    probs_g = F.sigmoid(per_frame_logits_g) * masks_gl.unsqueeze(1)
                else:
                    per_frame_logits = per_frame_logits.view(b,n,per_frame_logits.shape[1],tl)
                    probs = F.sigmoid(per_frame_logits) #* masks.unsqueeze(1)
                    probs = torch.max(probs, dim=1)[0] * masks.unsqueeze(1)
                    per_frame_logits = torch.max(per_frame_logits, dim=1)[0]

                    #logits_g = logits_g.view(bg,ng,logits_g.shape[1])
                    #logits_g = torch.max(logits_g, dim=1)[0]
                    per_frame_logits_g = per_frame_logits_g.view(bg,ng,per_frame_logits_g.shape[1],tl_g)
                    probs_g = F.sigmoid(per_frame_logits_g) #* masks.unsqueeze(1)
                    probs_g = torch.max(probs_g, dim=1)[0] * masks_gl.unsqueeze(1)
                    per_frame_logits_g = torch.max(per_frame_logits_g, dim=1)[0]

                #print(logits.shape, labels.shape)
                #cls_loss = criterion(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                cls_loss = criterion_class(torch.max(probs, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.item()

                #loc_loss = criterion(per_frame_logits, labels) # * input_mask.unsqueeze(1)
                loc_loss = criterion_loc(probs, labels) / (torch.sum(masks)*labels.shape[1])
                tot_loc_loss += loc_loss.item() #data[0]

                #cls_loss_g = criterion(torch.max(per_frame_logits_g, dim=2)[0], torch.max(labels_g, dim=2)[0])
                cls_loss_g = criterion_class(torch.max(probs_g, dim=2)[0], torch.max(labels_g, dim=2)[0])
                tot_cls_loss_g += cls_loss_g.item()

                #loc_loss_g = criterion(per_frame_logits_g, labels_g) # * input_mask.unsqueeze(1)
                loc_loss_g = criterion_loc(probs_g, labels_g) / (torch.sum(masks)*labels.shape[1])
                tot_loc_loss_g += loc_loss_g.item() #data[0]

                # Calculate accuracy
                #correct = torch.sum(preds == labels.data)
                #accuracy = correct.double() / logits.shape[0] #batch_size
                #tot_acc += accuracy
                #tot_corr += correct.double()
                #tot_dat += logits.shape[0]
                if phase == 'train':
                    for b in range(labels.shape[0]):
                        tr_apm.add(probs[b][:,:valid_t[b].item()].transpose(0,1).detach().cpu().numpy(),
                                    labels[b][:,:valid_t[b].item()].transpose(0,1).cpu().numpy())
                else:
                    for b in range(labels.shape[0]):
                        val_apm.add(probs[b][:,:valid_t[b].item()].transpose(0,1).detach().cpu().numpy(),
                                    labels[b][:,:valid_t[b].item()].transpose(0,1).cpu().numpy())


                # Calculate elapsed time for this step
                #examples_per_second = 64/float(time.time() - start_time)

                # Back-propagation and optimization step
                #loss.backward()
                #optimizer.step()

                # Save statistics
                #accuracies[step] = accuracy.item()
                #losses[step] = loss.item()

                loss = (cls_loss + loc_loss)/(2 * num_steps_per_update) + (cls_loss_g + loc_loss_g)/(2 * num_steps_per_update)
                #loss = 1 * (cls_loss + loc_loss)/(2 * num_steps_per_update) + cls_loss_g/num_steps_per_update
                #loss = 1 * (loc_loss)/(num_steps_per_update)

                #loss = (cls_loss + loc_loss)/(2 * num_steps_per_update)
                #loss = (cls_loss + loc_loss)/(2 * num_steps_per_update) + (cls_loss_g + loc_loss_g)/(2 * num_steps_per_update)
                tot_loss += loss.item() #data[0]

                if phase == 'train':
                    loss.backward()
                    '''
                    for name, para in i3d.named_parameters():
                        if '_rw' in name and para.grad is not None:
                            para.grad *= 5
                    '''
                            #print(name,para.grad.view(-1).detach().cpu().numpy()[:5])
                            #print('name',para.grad)
                        #if 'attnt' in name:
                        #    para.grad *= 5
                        #elif 'attn' in name:
                        #    para.grad *= 3

                if num_iter == num_steps_per_update and phase == 'train':
                    #lr_warmup(lr, steps-st_steps, warmup_steps, optimizer) # steps init_lr, USE ONLY AT THE START
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    #lr_sched.step()
                    s_times = iterations_per_epoch//2#100*4
                    if steps % s_times == 0: #(steps-load_steps) % s_times == 0:
                        #tot_acc = tot_corr/tot_dat
                        tr_map = tr_apm.value().mean()
                        tr_apm.reset()
                        #i3d.module.get_attn_para() #### print mu, sigma
                        print (' Epoch:{} {} steps: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Loc Loss G: {:.4f} Cls Loss G: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}'.format(epochs, phase,
                            steps, tot_loc_loss/(s_times*num_steps_per_update), tot_cls_loss/(s_times*num_steps_per_update),
                            tot_loc_loss_g/(s_times*num_steps_per_update), tot_cls_loss_g/(s_times*num_steps_per_update), tot_loss/s_times, tr_map))#, tot_acc/(s_times*num_steps_per_update)))
                        # save model
                        #torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = tot_loc_loss_g = tot_cls_loss_g = tot_dis_loss = tot_acc = tot_corr = tot_dat = 0.
                    if steps % (1000) == 0:
                        #tr_apm.reset()
                        if (not args.multiprocessing_distributed and args.rank == 0) or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                            ckpt = {'model_l_state_dict': x3d.module.state_dict(),
                                    'model_g_state_dict': x3d_g.module.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': lr_sched.state_dict()}
                            torch.save(ckpt, save_model+str(steps).zfill(6)+'.pt')
                        #torch.save(x3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        #torch.save(i3d2.module.state_dict(), save_model2+str(steps).zfill(6)+'.pt')
            if phase == 'val':
                val_map = val_apm.value().mean()
                #lr_sched.step(tot_loss)
                #lr_sched.step(tot_loc_loss)
                val_apm.reset()
                #tot_acc = tot_corr/tot_dat
                print (' Epoch:{} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Loc Loss G: {:.4f} Cls Loss G: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}'.format(epochs, phase,
                    tot_loc_loss/num_iter, tot_cls_loss/num_iter, tot_loc_loss_g/num_iter, tot_cls_loss_g/num_iter, (tot_loss*num_steps_per_update)/num_iter, val_map))#, tot_acc/num_iter))
                tot_loss = tot_loc_loss = tot_cls_loss = tot_loc_loss_g = tot_cls_loss_g = tot_dis_loss = tot_acc = tot_corr = tot_dat = 0.
                lr_sched.step()
                #lr_sched.step(tot_loss)


def init_cropping_scales(num_scales, init_scale, factor):
    # Determine cropping scales
    scales = [init_scale]
    for i in range(1, num_scales):
        scales.append(scales[-1] * factor)
    return scales

def lr_warmup(init_lr, cur_steps, warmup_steps, opt):
    start_after = 1
    if cur_steps < warmup_steps and cur_steps > start_after:
        lr_scale = min(1., float(cur_steps + 1) / warmup_steps)
        for pg in opt.param_groups:
            pg['lr'] = lr_scale * init_lr

def print_stats(long_ind, batch_size, stats, gamma_tau, bn_splits, lr):
    bs = batch_size * LONG_CYCLE[long_ind]
    if long_ind in [0,1]:
        bs = [bs*j for j in [2,1]]
        print(' ***** LR {} Frames {}/{} BS ({},{}) W/H ({},{}) BN_splits {} long_ind {} *****'.format(lr, stats[0][0], gamma_tau, bs[0], bs[1], stats[2][0], stats[3][0], bn_splits, long_ind))
    else:
        bs = [bs*j for j in [4,2,1]]
        print(' ***** LR {} Frames {}/{} BS ({},{},{}) W/H ({},{},{}) BN_splits {} long_ind {} *****'.format(lr, stats[0][0], gamma_tau, bs[0], bs[1], bs[2], stats[1][0], stats[2][0], stats[3][0], bn_splits, long_ind))

if __name__ == '__main__':
    # need to add argparse
    #run(mode=args.mode, root=args.root) #, save_model=args.save_model)
    args = parser.parse_args()
    main(args)
