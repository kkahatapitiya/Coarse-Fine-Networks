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


def load_rgb_frames(image_dir, fine_feat, feature_keys, vid, start, num, stride, video_loader):

  frame_indices = list(range(start, start+num, stride))
  frames = video_loader(image_dir, vid, frame_indices)

  keys = feature_keys
  feat = {}
  for k in keys:
      f = torch.load(os.path.join(fine_feat, k, vid)).squeeze(0) # C T 1 1
      if k == 'gx':f = f.view(1,-1,1,1) # 1 T 1 1
      feat[k] = f.numpy()

  return frames, feat


def make_dataset(split_file, split, root, num_classes=157):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    pre_data_file = split_file[:-5]+'_'+split+'labeldata_160.npy' #labeldata_160
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

    def __init__(self, split_file, split, root, fine_feat, feature_keys, spatial_transform=None, task='class', frames=80, gamma_tau=5, crops=1):

        self.data = make_dataset(split_file, split, root)
        self.split_file = split_file
        self.root = root
        self.frames = frames * 2
        self.gamma_tau = gamma_tau * 2 #2
        self.loader = get_default_video_loader()
        self.spatial_transform = spatial_transform
        self.crops = crops
        self.split = split
        self.task = task
        self.fine_feat = fine_feat
        self.feature_keys = feature_keys

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, dur, nf = self.data[index]

        if self.split == 'testing':
            frames = nf
            start_f = 1
        else:
            frames = min(self.frames, nf)
            start_f = random.randint(1, max(self.gamma_tau, nf-frames))

        stride_f = self.gamma_tau
        if self.split == 'testing' and self.task == 'loc':
            stride_f = stride_f//self.crops

        imgs, feat = load_rgb_frames(self.root, self.fine_feat, self.feature_keys, vid, start_f, frames, stride_f, self.loader) #stride_f


        label = label[:, start_f-1:start_f-1+frames:1] #stride_f
        label = torch.from_numpy(label)
        if self.task == 'class':
            label = torch.max(label, dim=1)[0] # C T --> C

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters(224)
            imgs_l = [self.spatial_transform(img) for img in imgs]
        imgs_l = torch.stack(imgs_l, 0).permute(1, 0, 2, 3) # T C H W --> C T H W

        step = 1
        if self.split == 'testing': #self.crops > 1:
            if self.task == 'class':
                step = int((imgs_l.shape[1] - 1 - self.frames//self.gamma_tau)//(self.crops-1))
                if step == 0:
                    clips = [imgs_l[:,:self.frames//self.gamma_tau,...] for i in range(self.crops)]
                    clips = torch.stack(clips, 0)
                else:
                    clips = [imgs_l[:,i:i+self.frames//self.gamma_tau,...] for i in range(0, step*self.crops, step)]
                    clips = torch.stack(clips, 0)
            if self.task == 'loc': #self.crops > 1:
                clips = [imgs_l[:,i::self.crops,...][:,:frames//self.gamma_tau,...] for i in range(0, self.crops)]
                clips = torch.stack(clips, 0) # N C T H W
                label = label[:,:(frames//self.gamma_tau)*self.gamma_tau]
        else:
            clips = imgs_l.unsqueeze(0) # 1 C T H W

        meta = torch.from_numpy(np.array([start_f//self.gamma_tau, frames//self.gamma_tau,
                                            nf//self.gamma_tau, stride_f//self.gamma_tau]))

        return clips, label, feat, meta, vid, dur #feat_conv5

    def __len__(self):
        return len(self.data)


def mt_collate_fn(batch):
    "Pads data and puts it into a tensor of same dimensions"
    cap = 128 # NEW LIMIT FOR XYTC MIXING
    #cap_clip = 512 # for testing
    #cap_label = cap_clip*10

    max_len_clips = 0
    max_len_labels = 0
    max_len_feat =  0 #cap
    for b in batch:
        if b[0].shape[2] > max_len_clips:
            max_len_clips = b[0].shape[2]
        if b[1].shape[1] > max_len_labels:
            max_len_labels = b[1].shape[1]
        if list(b[2].values())[0].shape[1] > max_len_feat:
            max_len_feat = list(b[2].values())[0].shape[1]
    max_len_feat = min(max_len_feat, cap)
    #max_len_clips = min(max_len_clips, cap_clip)
    #max_len_labels = min(max_len_labels, cap_label)

    new_batch = []
    #c,t,h,w = list(batch[0][2].values())[0].shape
    keys = list(batch[0][2].keys())
    for b in batch:
        clips = np.zeros((b[0].shape[0], b[0].shape[1], max_len_clips, b[0].shape[3], b[0].shape[4]), np.float32)
        label = np.zeros((b[1].shape[0], max_len_labels), np.float32)
        mask = np.zeros((max_len_labels), np.float32)
        feat_mask = np.zeros((max_len_feat), np.float32)

        clips[:,:,:b[0].shape[2],:,:] = b[0]#[:,:,:min(cap_clip,b[0].shape[2]),:,:]
        label[:,:b[1].shape[1]] = b[1]#[:,:min(cap_label,b[1].shape[1])]
        mask[:b[1].shape[1]] = 1

        feat_mask[:min(cap,list(b[2].values())[0].shape[1])] = 1
        feat = {}
        for k in keys:
            c,t,h,w = b[2][k].shape
            f = np.zeros((c, max_len_feat, h, w), np.float32)
            f[:,:min(cap,b[2][k].shape[1]),:,:] = b[2][k][:,:min(cap,b[2][k].shape[1]),:,:]
            feat[k] = torch.from_numpy(f)

        new_batch.append([torch.from_numpy(clips), torch.from_numpy(label), torch.from_numpy(mask),
                        feat, torch.from_numpy(feat_mask), b[3], b[4], b[5]])

    return default_collate(new_batch)
