import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path
import functools

import torchvision
#import torch.utils.data as data
from PIL import Image

import cv2

random.seed(0)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    torchvision.set_image_backend('accimage')
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, vid, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, vid, vid+'-'+str(i).zfill(6)+'.jpg')
        #image_path = os.path.join(video_dir_path, 'frame_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, x3d_out, feature_keys, vid, start, num, stride, video_loader, load_feat=False):
  #frames = []

  frame_indices = list(range(start, start+num, stride))
  frames = video_loader(image_dir, vid, frame_indices)

  #feat_conv5 = torch.load(os.path.join(x3d_out['conv5'], vid)).numpy() # N C T 1 1
  if load_feat:
      keys = feature_keys
      feat = {}
      for k in keys:
          f = torch.load(os.path.join(x3d_out[0], k, vid)).squeeze(0) # C T 1 1
          if k == 'gx':f = f.view(1,-1,1,1) # 1 T 1 1
          feat[k] = f.numpy()

      return frames, feat

  '''for i in range(start, start+num, stride):
    img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < resize_size or h < resize_size:
        d = float(resize_size)-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  frames = np.asarray(frames, dtype=np.float32)'''
  return frames#, feat_conv5

def load_flow_frames(image_dir, vid, start, num, resize_size):
  frames = []
  for i in range(start, start+num):
    imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)

    w,h = imgx.shape
    if w < resize_size or h < resize_size:
        d = float(resize_size)-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)

    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=157):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    pre_data_file = split_file[:-5]+'_'+split+'labeldata_160.npy'
    if os.path.exists(pre_data_file):
        print('{} exists'.format(pre_data_file))
        dataset = np.load(pre_data_file, allow_pickle=True)
    else:
        print('{} does not exist'.format(pre_data_file))
        i = 0
        for vid in data.keys():
            if data[vid]['subset'] != split:
                continue

            if not os.path.exists(os.path.join(root, vid)):
                continue
            num_frames = len(os.listdir(os.path.join(root, vid)))
            if mode == 'flow':
                num_frames = num_frames//2

            if num_frames < (2*80+2):
                continue

            label = np.zeros((num_classes,num_frames), np.float32)

            fps = num_frames/data[vid]['duration']
            for ann in data[vid]['actions']:
                for fr in range(0,num_frames,1):
                    if fr/fps > ann[1] and fr/fps < ann[2]:
                        label[ann[0], fr] = 1 # binary classification
            dataset.append((vid, label, data[vid]['duration'], num_frames))
            i += 1
            print(i, vid)
        np.save(pre_data_file, dataset)

    print('dataset size:%d'%len(dataset))
    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, x3d_out, feature_keys, mode, spatial_transform=None, task='class', frames=80, gamma_tau=5, crops=1):

        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.mode = mode
        self.root = root
        self.frames = frames * 2
        self.gamma_tau = gamma_tau * 2 #2
        self.loader = get_default_video_loader()
        self.spatial_transform = spatial_transform
        #self.val_spatial_transform = val_spatial_transform
        self.crops = crops
        self.split = split
        self.task = task
        self.x3d_out = x3d_out
        self.feature_keys = feature_keys
        #self.const_frames = 32
        self.cap = 128 #128
        self.train_cap = 64 #128

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label_in, dur, nf = self.data[index]
        if self.split == 'testing':
            frames = nf
            start_f = 1
            #frames_g = nf
            #start_f_g = 1

            #frames_g = nf
            #stride_f_g = nf//self.const_frames
            #start_f_g = 1
            frames_g = min(nf, self.cap*self.gamma_tau)
            stride_f_g = self.gamma_tau
            start_f_g = max(1, (nf - frames_g)//2)
        else:
            #frames = self.frames * 4
            #start_f = random.randint(1,nf-(self.frames+1))
            frames = min(self.frames * 4, nf)
            start_f = random.randint(1, max(self.gamma_tau, nf-frames))
            #start_f_g = min(start_f, random.randint(1,max(nf-(64*self.gamma_tau*2 + 1), self.gamma_tau*2)))#1
            #frames_g = min(nf - start_f_g, 64*self.gamma_tau*2)

            #frames_g = nf
            #stride_f_g = nf//self.const_frames
            #start_f_g = random.randint(1,max(1,nf-stride_f_g*self.const_frames-1))
            frames_g = min(nf, self.train_cap*self.gamma_tau)
            stride_f_g = self.gamma_tau
            start_f_g = random.randint(1, min(start_f, max(stride_f_g, nf-frames_g)))

        stride_f = self.gamma_tau
        #stride_f_g = self.gamma_tau * 2 # 20
        '''if self.split == 'testing' and self.task == 'loc':
            stride_f = stride_f//self.crops
            stride_f_g = stride_f_g//self.crops'''

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, self.x3d_out, self.feature_keys, vid, start_f, frames,
                                        stride_f, self.loader, load_feat=False) #stride_f
            imgs2 = load_rgb_frames(self.root, self.x3d_out, self.feature_keys, vid, start_f_g, frames_g,
                                        stride_f_g, self.loader) # imgs if self.split == 'testing' else
        else:
            imgs = load_flow_frames(self.root, vid, start_f, self.frames)


        label = label_in[:, start_f-1:start_f-1+frames:1] #stride_f
        label2 = label_in[:, start_f_g-1:start_f_g-1+frames_g:1] # label if self.split == 'testing' else
        label = torch.from_numpy(label)
        label2 = torch.from_numpy(label2)
        #label2 = torch.max(label2, dim=1)[0] # C T --> C

        #if self.task == 'class':
        #    label = torch.max(label, dim=1)[0] # C T --> C

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters(224)
            imgs_l = [self.spatial_transform(img) for img in imgs]
            imgs_g = [self.spatial_transform(img) for img in imgs2] # imgs_l if self.split == 'testing' else
        imgs_l = torch.stack(imgs_l, 0).permute(1, 0, 2, 3) # T C H W --> C T H W
        imgs_g = torch.stack(imgs_g, 0).permute(1, 0, 2, 3) # imgs_l if self.split == 'testing' else
        #imgs = self.transforms(imgs)
        #print(imgs.shape, label.shape)


        '''if self.split == 'testing':
            #if self.task == 'class':
                #step = int((imgs_l.shape[1] - 1 - self.frames//self.gamma_tau)//(self.crops-1))
                #if step == 0:
                    #clips = [imgs_l[:,:self.frames//self.gamma_tau,...] for i in range(self.crops)]
                    #clips = torch.stack(clips, 0)
                #else:
                    #clips = [imgs_l[:,i:i+self.frames//self.gamma_tau,...] for i in range(0, step*self.crops, step)]
                    #clips = torch.stack(clips, 0)
            if self.task == 'loc':
                clips = [imgs_l[:,i::self.crops,...][:,:frames//self.gamma_tau,...] for i in range(0, self.crops)]
                clips = torch.stack(clips, 0) # N C T H W
                clips2 = [imgs_g[:,i::self.crops,...][:,:frames_g//(self.gamma_tau*1),...] for i in range(0, self.crops)]
                #clips2 = [imgs_g[:,i::self.crops,...][:,:self.const_frames,...] for i in range(0, self.crops)]
                clips2 = torch.stack(clips2, 0) # N C T H W
        else:'''
        clips = imgs_l.unsqueeze(0) # 1 C T H W
        #clips2 = imgs_g.unsqueeze(0) # 1 C T H W
        #clips2 = imgs_g[:,:self.const_frames,...].unsqueeze(0) # 1 C T H W
        clips2 = imgs_g.unsqueeze(0) # 1 C T H W

        #n,c,t,h,w = feat_conv5.shape
        ##feat_conv5 = torch.from_numpy(feat_conv5).permute(1,2,0,3,4).reshape(c,t*n,h,w) # C TN 1 1
        #feat_conv5 = torch.from_numpy(feat_conv5[0]) # C T 1 1

        meta = torch.from_numpy(np.array([(start_f-start_f_g)//self.gamma_tau, frames//self.gamma_tau,
                                            nf//self.gamma_tau, stride_f//self.gamma_tau]))


        return clips, label, clips2, label2, meta, vid, dur

    def __len__(self):
        return len(self.data)


def mt_collate_fn(batch):
    "Pads data and puts it into a tensor of same dimensions"
    #cap = 128 #32 #64 #128
    #capl = 128*10 #64 * 20 #128 * 5 * 2

    max_len_clips = 0
    max_len_labels = 0
    max_len_clips2 =  0 #cap
    max_len_labels2 =  0 #capl
    #max_len_feat = cap

    for b in batch:
        if b[0].shape[2] > max_len_clips:
            max_len_clips = b[0].shape[2]
        if b[1].shape[1] > max_len_labels:
            max_len_labels = b[1].shape[1]

        if b[2].shape[2] > max_len_clips2:
            max_len_clips2 = b[2].shape[2]
        if b[3].shape[1] > max_len_labels2:
            max_len_labels2 = b[3].shape[1]

        '''if list(b[4].values())[0].shape[1] > max_len_feat:
            max_len_feat = list(b[4].values())[0].shape[1]'''

    #max_len_clips2 = min(max_len_clips2, cap)
    #max_len_labels2 = min(max_len_labels2, capl)
    #max_len_feat = min(max_len_feat, cap)

    new_batch = []
    #keys = list(batch[0][4].keys())
    for b in batch:
        clips = np.zeros((b[0].shape[0], b[0].shape[1], max_len_clips, b[0].shape[3], b[0].shape[4]), np.float32)
        label = np.zeros((b[1].shape[0], max_len_labels), np.float32)
        mask = np.zeros((max_len_labels), np.float32) # label mask, no striding

        clips2 = np.zeros((b[2].shape[0], b[2].shape[1], max_len_clips2, b[2].shape[3], b[2].shape[4]), np.float32)
        label2 = np.zeros((b[3].shape[0], max_len_labels2), np.float32)
        #mask2 = np.zeros((max_len_clips2), np.float32) # clip mask, chnaged to label mask
        mask2 = np.zeros((max_len_labels2), np.float32)

        '''feat_mask = np.zeros((max_len_feat), np.float32)
        feat = {}'''

        clips[:,:,:b[0].shape[2],:,:] = b[0]
        label[:,:b[1].shape[1]] = b[1]
        mask[:b[1].shape[1]] = 1

        #clips2[:,:,:min(cap,b[2].shape[2]),:,:] = b[2][:,:,:min(cap,b[2].shape[2]),:,:]
        #label2[:,:min(capl,b[3].shape[1])] = b[3][:,:min(capl,b[3].shape[1])]
        ##mask2[:min(cap,b[2].shape[2])] = 1
        #mask2[:min(capl,b[3].shape[1])] = 1
        clips2[:,:,:b[2].shape[2],:,:] = b[2]
        label2[:,:b[3].shape[1]] = b[3]
        mask2[:b[3].shape[1]] = 1

        '''feat_mask[:min(cap,list(b[4].values())[0].shape[1])] = 1
        for k in keys:
            c,t,h,w = b[4][k].shape
            f = np.zeros((c, max_len_feat, h, w), np.float32)
            f[:,:min(cap,b[4][k].shape[1]),:,:] = b[4][k][:,:min(cap,b[4][k].shape[1]),:,:]
            feat[k] = torch.from_numpy(f)'''

        new_batch.append([torch.from_numpy(clips), torch.from_numpy(label), torch.from_numpy(mask),
                        torch.from_numpy(clips2), torch.from_numpy(label2), torch.from_numpy(mask2),
                        b[4], b[5], b[6]])

    return default_collate(new_batch)
