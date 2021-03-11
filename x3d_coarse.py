import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from interp1d import Interp1d


class SubBatchNorm3d(nn.Module):
    """ FROM SLOWFAST """
    def __init__(self, num_splits, **args):
        super(SubBatchNorm3d, self).__init__()
        self.num_splits = num_splits
        self.num_features = args["num_features"]
        # Keep only one set of weight and bias.
        if args.get("affine", True):
            self.affine = True
            args["affine"] = False
            self.weight = torch.nn.Parameter(torch.ones(self.num_features))
            self.bias = torch.nn.Parameter(torch.zeros(self.num_features))
        else:
            self.affine = False
        self.bn = nn.BatchNorm3d(**args)
        args["num_features"] = self.num_features * self.num_splits
        self.split_bn = nn.BatchNorm3d(**args)

    def _get_aggregated_mean_std(self, means, stds, n):
        mean = means.view(n, -1).sum(0) / n
        std = (
            stds.view(n, -1).sum(0) / n
            + ((means.view(n, -1) - mean) ** 2).view(n, -1).sum(0) / n
        )
        return mean.detach(), std.detach()

    def aggregate_stats(self):
        """Synchronize running_mean, and running_var. Call this before eval."""
        if self.split_bn.track_running_stats:
            (
                self.bn.running_mean.data,
                self.bn.running_var.data,
            ) = self._get_aggregated_mean_std(
                self.split_bn.running_mean,
                self.split_bn.running_var,
                self.num_splits,
            )

    def forward(self, x):
        if self.training:
            n, c, t, h, w = x.shape
            x = x.view(n // self.num_splits, c * self.num_splits, t, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, t, h, w)
        else:
            x = self.bn(x)
        if self.affine:
            x = x * self.weight.view((-1, 1, 1, 1))
            x = x + self.bias.view((-1, 1, 1, 1))
        return x


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return SwishEfficient.apply(x)


class SwishEfficient(torch.autograd.Function):
    """Swish activation function: x * sigmoid(x)."""
    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=(1,stride,stride),
                     padding=1,
                     bias=False,
                     groups=in_planes
                     )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=(1,stride,stride),
                     bias=False)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None, index=0, base_bn_splits=8):
        super(Bottleneck, self).__init__()

        self.index = index
        self.base_bn_splits = base_bn_splits
        self.conv1 = conv1x1x1(in_planes, planes[0])
        self.bn1 = SubBatchNorm3d(num_splits=self.base_bn_splits, num_features=planes[0], affine=True) #nn.BatchNorm3d(planes[0])
        self.conv2 = conv3x3x3(planes[0], planes[0], stride)
        self.bn2 = SubBatchNorm3d(num_splits=self.base_bn_splits, num_features=planes[0], affine=True) #nn.BatchNorm3d(planes[0])
        self.conv3 = conv1x1x1(planes[0], planes[1])
        self.bn3 = SubBatchNorm3d(num_splits=self.base_bn_splits, num_features=planes[1], affine=True) #nn.BatchNorm3d(planes[1])
        self.swish = Swish() #nn.Hardswish()
        self.relu = nn.ReLU(inplace=True)
        if self.index % 2 == 0:
            width = self.round_width(planes[0])
            self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
            self.fc1 = nn.Conv3d(planes[0], width, kernel_size=1, stride=1)
            self.fc2 = nn.Conv3d(width, planes[0], kernel_size=1, stride=1)
            self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def round_width(self, width, multiplier=0.0625, min_width=8, divisor=8):
        if not multiplier:
            return width

        width *= multiplier
        min_width = min_width or divisor
        width_out = max(
            min_width, int(width + divisor / 2) // divisor * divisor
        )
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Squeeze-and-Excitation
        if self.index % 2 == 0:
            se_w = self.global_pool(out)
            se_w = self.fc1(se_w)
            se_w = self.relu(se_w)
            se_w = self.fc2(se_w)
            se_w = self.sigmoid(se_w)
            out = out * se_w
        out = self.swish(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RewightLayer(nn.Module):
    def __init__(self, channels, g_channels, depth, height, pool=False):
        super(RewightLayer, self).__init__()

        fc_depth = depth
        attn_depth = depth #//2

        self.at1 = nn.Conv1d(depth, attn_depth, kernel_size=1, stride=1, padding=0)
        self.at2 = nn.Conv1d(attn_depth, 1, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Conv1d(depth, fc_depth, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv1d(fc_depth, channels, kernel_size=1, stride=1, padding=0)
        if g_channels is not None:
            self.fc3 = nn.Conv1d(depth, fc_depth, kernel_size=1, stride=1, padding=0)
            self.fc4 = nn.Conv1d(fc_depth, g_channels, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(0.5)

        self.depth = depth
        self.height = height
        self.channels = channels
        self.g_channels = g_channels
        self.pool = pool


    def forward(self, inp):
        x, lx, mask, gx, i, GX, isMixing = inp # x: B C T H W ---- B Tbar
        b,c,t,h,w = x.shape
        b2,_,tl,_,_ = lx.shape
        hl,wl = self.height, self.height

        if mask.shape[1] != t:
            mask = F.adaptive_max_pool1d(mask.unsqueeze(1), t).squeeze(1)
            GX = F.adaptive_avg_pool2d(GX.unsqueeze(1), (t,None)).squeeze(1)

        if b != b2: # for multi-crop testing
            x = x.unsqueeze(1).repeat(1,b2//b,1,1,1,1).view(b2,c,t,h,w)
            mask = mask.unsqueeze(1).repeat(1,b2//b,1).view(b2,t)

        if h != hl: # for 7x7 feat inputs
            x = F.adaptive_max_pool2d(x.view(b2,c*t,h,w), (hl,wl)).view(b2,c,t,hl,wl)

        at = x.view(b2,c,-1)
        at = F.relu(self.at1(at), inplace=True)
        at = self.at2(at) # B 1 T H W
        at = F.sigmoid(at).view(b2,-1,t,hl,wl)

        at = at.unsqueeze(3) * GX.view(b2,1,t,tl,1,1)
        x = x.unsqueeze(3) * at # B C T  Tl Hl Wl
        mask = mask.view(b2,1,t,1,1,1) # B 1 T 1 1 1
        temp_weights = mask/(torch.sum(at*mask, dim=2, keepdim=True)+1e-6) # B 1 T Tl Hl Wl
        x = torch.sum(x * temp_weights, dim=2)#.unsqueeze(2) # B C Tl Hl Wl

        if self.pool: # NOT USED
            x = F.adaptive_avg_pool3d(x, (None,1,1))

        b,c,t,h,w = x.shape

        x1 = F.relu(self.fc1(x.view(b,c,-1)), inplace=True)
        if self.pool:# and not(isMixing): # NOT USED
            x1 = self.dropout(x1)
        x1 = self.fc2(x1)
        x1 = x1.view(b,-1,t,h,w) # B D 1 H W

        if self.g_channels is not None:
            x2 = F.relu(self.fc3(x.view(b,c,-1)), inplace=True)
            if self.pool:# and not(isMixing): # NOT USED
                x2 = self.dropout(x2)
            x2 = self.fc4(x2)
            x2 = x2.view(b,-1,t,h,w) # B D 1 H W
            if not(isMixing):
                x2 = F.sigmoid(x2)
            return x1, x2
        return x1



class Gaussian(nn.Module):
    def __init__(self, ratio=1):
        super(Gaussian, self).__init__()
        self.ratio = ratio #1

    def forward(self, inp):

        meta, mask, gx, tx = inp
        st, fr, nf, step = meta[:,0], meta[:,1], meta[:,2], meta[:,3]
        b = meta.shape[0]
        b2 = gx.shape[0]
        len_f = mask.shape[1]

        if b2 != b: # multi-crop testing
            offset = step.view(-1,1) * torch.arange(0,b2//b).to(torch.float32).view(1,-1).repeat(b,1).cuda() #0,...,10-1
            st = (st.view(-1,1).repeat(1,b2//b) + offset).view(-1,1)

        if tx is not None:
            len_x = gx.shape[1]
            tl = (gx * tx).unsqueeze(1)
        else:
            len_x = gx.shape[2]
            tl = torch.arange(0,len_x).to(torch.float32).view(1,1,-1).repeat(b2,1,1).cuda()
            #tl = 4* torch.arange(0,len_x).to(torch.float32).view(1,1,-1).repeat(b2,1,1).cuda() # STRIDING
        mu = (tl + st.view(b2,1,1))/self.ratio # B 1 Tl

        t = torch.arange(0,len_f).to(torch.float32).view(1,-1,1).repeat(b2,1,1).cuda() # independent var
        std = (1/8 * torch.sum(mask, dim=1)).view(-1,1).repeat(1,b2//b).view(-1,1) #1/8

        t = t - mu # B T Tl
        f = t**2 / (2 * (std**2).view(b2,1,1).repeat(1,len_f,len_x) + 1e-16)
        f = torch.exp(-f)
        f = f / (torch.max(f, dim=1)[0].view(b2,1,len_x) + 1e-16)
        f = f.view(b2,len_f,len_x)

        return f


class MixingLayer(nn.Module):
    def __init__(self, depth, learned=False, index=0, isLogit=False):
        super(MixingLayer, self).__init__()

        self.learned = learned
        self.index = index
        self.isLogit = isLogit

        self.in_depth = 432 if isLogit else (24+48+96+192)
        self.range = 1 if isLogit else 4

        self.dropout = nn.Dropout(0.5)

        if learned:
            self.conv_at = nn.Conv1d(self.in_depth, depth, kernel_size=1, stride=1, padding=0)
            self.conv_at2 = nn.Conv1d(self.in_depth, depth, kernel_size=1, stride=1, padding=0)


    def forward(self, inp):
        x, bias, scale = inp
        b,c,t,h,w = x.shape

        cs = []; ms = [];
        for i in range(self.range): # B C1 1 H1 W1
            _,cf,_,hf,wf = bias[i].shape
            if hf != h:
                bias_i = F.adaptive_max_pool2d(bias[i].view(b,cf*t,hf,wf), (h,w)).view(b,cf,t,h,w) # B 1 C1 H W
            else:
                bias_i = bias[i]
            cs.append(bias_i)
        for j in range(self.range):
            _,cf,_,hf,wf = scale[j].shape
            if hf != h:
                scale_j = F.adaptive_max_pool2d(scale[j].view(b,cf*t,hf,wf), (h,w)).view(b,cf,t,h,w)
            else:
                scale_j = scale[j]
            ms.append(scale_j)

        cs = torch.cat(cs, dim=1)#.detach() # B C+ T H W
        ms = torch.cat(ms, dim=1)#.detach()


        if self.learned:
            if self.isLogit:
                cs = self.dropout(cs)
                ms = self.dropout(ms)
            cs = self.conv_at(cs.view(b,-1,t*h*w)).view(b,c,t,h,w)
            ms = F.sigmoid(self.conv_at2(ms.view(b,-1,t*h*w))).view(b,c,t,h,w)

        else:
            l = [24,48,96,192].index(c)
            at = F.one_hot(torch.arange(4), 4).cuda()[l]
            at = at.view(1,1,-1).repeat(b,c,1) # B C 4

            cs = torch.sum(cs * at.unsqueeze(3).unsqueeze(4).unsqueeze(5), dim=2) # B C 1 H W
            ms = torch.sum(ms * at.unsqueeze(3).unsqueeze(4).unsqueeze(5), dim=2)

        '''bf,_,_,_,_ = cs.shape
        if bf != b: # FOR MULTI-CROP TESTING
            cs = cs.unsqueeze(1).repeat(1,b//bf,1,1,1,1).view(b,c,1,h,w)
            ms = ms.unsqueeze(1).repeat(1,b//bf,1,1,1,1).view(b,c,1,h,w)'''

        return cs, ms



class GridPoolLayer(nn.Module):
    def __init__(self, ratio, depth):
        super(GridPoolLayer, self).__init__()

        self.ratio = 4 #4 #ratio
        self.depth = depth

        self.conv1 = nn.Conv3d(depth, depth, kernel_size=(3,3,3), stride=(self.ratio//2,2,2), padding=1) # ratio//2
        self.bn1 = SubBatchNorm3d(num_splits=1, num_features=depth, affine=True)
        self.conv2 = nn.Conv3d(depth, depth, kernel_size=(3,3,3), stride=(self.ratio//2,2,2), padding=1) # ratio//2
        self.bn2 = SubBatchNorm3d(num_splits=1, num_features=depth, affine=True)
        self.conv3 = nn.Conv3d(depth, 1, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        #self.conv3 = nn.Conv3d(depth, 1, kernel_size=(3,3,3), stride=(ratio//4,2,2), padding=(1,1,1)) #(1,3,3), (1,2,2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        #self.pool_optional = nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))


    def forward(self, inp):
        x = inp
        #x = inp
        b,c,t,h,w = x.shape
        x = x#.detach()

        gx = self.relu(self.bn1(self.conv1(x)))
        gx = self.relu(self.bn2(self.conv2(gx)))
        gx = self.conv3(gx) # B 1 32 7 7
        gx = F.adaptive_avg_pool3d(gx, (None,1,1))
        gx = gx.squeeze(4).squeeze(3).squeeze(1) # B 32
        gx = self.sigmoid(gx * 5e-1)

        gx = 1. - gx
        gx = gx/(torch.sum(gx, dim=1, keepdim=True)+1e-16)
        gx = torch.cumsum(gx, dim=1)
        #print(' ',t, gx[0]*t)
        gx_ = torch.zeros(gx.shape[0], gx.shape[1]+1).to(torch.float32).cuda()
        gx_[:,1:] = gx
        gx_out = gx_

        gx = (gx_ - 0.5) * 2 # B 32+1

        gh = torch.arange(h).to(torch.float32).cuda() / (h-1)
        gw = torch.arange(w).to(torch.float32).cuda() / (w-1)
        gh = (gh - 0.5) * 2
        gw = (gw - 0.5) * 2
        grid = torch.meshgrid([gx.view(-1), gh, gw])
        grid = torch.stack((grid[2], grid[1], grid[0]), dim=-1).view(b,gx.shape[1],gh.shape[0],gw.shape[0],3) # B 32 H W 3

        x = F.grid_sample(x, grid, align_corners=True) # B C 32 H W
        #x = F.avg_pool3d(x, kernel_size=(4,1,1), stride=(4,1,1)) # B C 32 H W
        #x = self.pool_optional(x)

        '''mh = torch.zeros(1).to(torch.float32).cuda()
        mw = torch.zeros(1).to(torch.float32).cuda()
        grid_mask = torch.meshgrid([gx.view(-1), mh, mw])
        grid_mask = torch.stack((grid_mask[2], grid_mask[1], grid_mask[0]), dim=-1).view(b,gx.shape[1],mh.shape[0],mw.shape[0],3) # B 32+1 H W 3
        masks = masks.unsqueeze(1).unsqueeze(3).unsqueeze(4) # B 1 128 1 1
        masks = F.grid_sample(masks, grid_mask, mode='nearest', align_corners=True)
        #masks = F.max_pool3d(masks, kernel_size=(4,1,1), stride=(4,1,1))
        masks = masks.squeeze(4).squeeze(3).squeeze(1) # B 32'''

        return x, gx_out#, masks


def GridUnpool(inp):
    x, gx, is_logit = inp # inverse temporal grid here
    ratio = 4 #4

    if is_logit:
        b,c,t = x.shape
        x = x.unsqueeze(3).unsqueeze(4) # B C T 1 1
        gh = torch.zeros(1).to(torch.float32).cuda()
        gw = torch.zeros(1).to(torch.float32).cuda()
    else:
        b,c,t,h,w = x.shape
        gh = torch.arange(h).to(torch.float32).cuda() / (h-1)
        gw = torch.arange(w).to(torch.float32).cuda() / (w-1)
        gh = (gh - 0.5) * 2
        gw = (gw - 0.5) * 2

    mid = torch.arange(gx.shape[1]).to(torch.float32).cuda()
    mid = mid/(mid.shape[0]-1.)
    mid = mid.view(1,-1).repeat(b,1)
    gx_ = Interp1d()(gx, mid, mid, None)

    gx = (gx_ - 0.5) * 2 # B 32+1

    grid = torch.meshgrid([gx.view(-1), gh, gw])
    grid = torch.stack((grid[2], grid[1], grid[0]), dim=-1).view(b,gx.shape[1],gh.shape[0],gw.shape[0],3) # B 32+1 H W 3

    x = F.grid_sample(x, grid, align_corners=True) # B C 32 1 1
    if is_logit:
        x = x.squeeze(4).squeeze(3)
    else:
        x = F.interpolate(x, (t*ratio,h,w), mode='trilinear', align_corners=True)

    return x



class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 feat_depth={},
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 shortcut_type='B',
                 widen_factor=1.0,
                 dropout=0.5,
                 n_classes=400,
                 base_bn_splits=8,
                 task='class',
                 extract_feat=False,
                 t_pool=None,
                 learnedMixing=False,
                 isMixing=False):
        super(ResNet, self).__init__()

        block_inplanes = [(int(x * widen_factor),int(y * widen_factor)) for x,y in block_inplanes]
        self.index = 0
        self.base_bn_splits = base_bn_splits
        self.task = task
        self.extract_feat = extract_feat
        self.feat_depth = feat_depth
        self.learnedMixing = learnedMixing
        self.isMixing = isMixing
        self.t_pool = t_pool

        self.in_planes = block_inplanes[0][1]

        if self.t_pool == 'avg':
            self.pool_1 = nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))
        elif self.t_pool == 'max':
            self.pool_1 = nn.MaxPool3d((4, 1, 1), stride=(4, 1, 1))
        elif self.t_pool == 'grid':
            self.pool_1 = GridPoolLayer(ratio=4, depth=block_inplanes[0][1])

        self.conv1_s = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(1, 3, 3),
                               stride=(1, 2, 2),
                               padding=(0, 1, 1),
                               bias=False)
        self.conv1_t = nn.Conv3d(self.in_planes,
                               self.in_planes,
                               kernel_size=(5, 1, 1),
                               stride=(1, 1, 1),
                               padding=(2, 0, 0),
                               bias=False,
                               groups=self.in_planes)
        self.bn1 = SubBatchNorm3d(num_splits=self.base_bn_splits, num_features=self.in_planes, affine=True) #nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block,
                                       block_inplanes[0],
                                       layers[0],
                                       shortcut_type,
                                       stride=2)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        self.conv5 = nn.Conv3d(block_inplanes[3][1],block_inplanes[3][0],kernel_size=(1, 1, 1),stride=(1, 1, 1),padding=(0, 0, 0),bias=False)
        self.bn5 = SubBatchNorm3d(num_splits=self.base_bn_splits, num_features=block_inplanes[3][0], affine=True) #nn.BatchNorm3d(block_inplanes[3][0])


        self.rw2 = RewightLayer(channels=block_inplanes[0][1], g_channels=block_inplanes[0][1], depth=self.feat_depth['layer1'], height=56)
        self.rw3 = RewightLayer(channels=block_inplanes[1][1], g_channels=block_inplanes[1][1], depth=self.feat_depth['layer2'], height=28)
        self.rw4 = RewightLayer(channels=block_inplanes[2][1], g_channels=block_inplanes[2][1], depth=self.feat_depth['layer3'], height=14)
        self.rw5 = RewightLayer(channels=block_inplanes[3][1], g_channels=block_inplanes[3][1], depth=self.feat_depth['layer4'], height=7)
        self.rw6 = RewightLayer(channels=157, g_channels=157, depth=self.feat_depth['conv5'], height=7, pool=True)

        if self.isMixing:
            self.mix2 = MixingLayer(depth=block_inplanes[0][1], learned=self.learnedMixing, index=0)
            self.mix3 = MixingLayer(depth=block_inplanes[1][1], learned=self.learnedMixing, index=1)
            self.mix4 = MixingLayer(depth=block_inplanes[2][1], learned=self.learnedMixing, index=2)
            self.mix5 = MixingLayer(depth=block_inplanes[3][1], learned=self.learnedMixing, index=3)

        self.gauss = Gaussian(ratio=1) ####### 4 FOR GRIDPOOL

        if task == 'class':
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif task == 'loc':
            self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc1 = nn.Conv3d(block_inplanes[3][0], 2048, bias=False, kernel_size=1, stride=1) #nn.Linear(block_inplanes[3][0], 2048, bias=False)
        self.fc2 = nn.Linear(2048, n_classes)
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes[1]:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes[1],
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes[1], stride),
                    SubBatchNorm3d(num_splits=self.base_bn_splits, num_features=planes[1], affine=True) #nn.BatchNorm3d(planes[1])
                    )

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  index=self.index,
                  base_bn_splits=self.base_bn_splits))
        self.in_planes = planes[1]
        self.index += 1
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, index=self.index, base_bn_splits=self.base_bn_splits))
            self.index += 1

        self.index = 0
        return nn.Sequential(*layers)


    def replace_logits(self, n_classes):
        self.fc2 = nn.Linear(2048, n_classes)
        self.rw6 = RewightLayer(channels=n_classes, g_channels=n_classes, depth=self.feat_depth['conv5'], height=7, pool=True)


    def update_bn_splits_long_cycle(self, long_cycle_bn_scale):
        for m in self.modules():
            if isinstance(m, SubBatchNorm3d):
                m.num_splits = self.base_bn_splits * long_cycle_bn_scale
                m.split_bn = nn.BatchNorm3d(num_features=m.num_features*m.num_splits, affine=False).to(m.weight.device)
        return self.base_bn_splits * long_cycle_bn_scale


    def aggregate_sub_bn_stats(self):
        """find all SubBN modules and aggregate sub-BN stats."""
        count = 0
        for m in self.modules():
            if isinstance(m, SubBatchNorm3d):
                m.aggregate_stats()
                count += 1
        return count


    def forward(self, inp):
        x, feat, feat_masks, i, meta = inp
        _,_,tl,_,_ = x.shape


        x = self.conv1_s(x)
        x = self.conv1_t(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)

        if self.t_pool == 'avg':
            x = self.pool_1(x)
        elif self.t_pool == 'max':
            x = self.pool_1(x)
        elif self.t_pool == 'stride':
            x = x[:,:,::4,:,:]
        elif self.t_pool == 'grid':
            x_up = x
            x, gx = self.pool_1(x)
            x_down = x
            GX = self.gauss([meta, feat_masks, gx, tl])

        if self.t_pool != 'grid':
            GX = self.gauss([meta, feat_masks, x, None])

        if self.isMixing:
            rw2, rw2_g = self.rw2([feat['layer1'], x, feat_masks, None, i, GX, True])
            rw3, rw3_g = self.rw3([feat['layer2'], x, feat_masks, None, i, GX, True])
            rw4, rw4_g = self.rw4([feat['layer3'], x, feat_masks, None, i, GX, True])
            rw5, rw5_g = self.rw5([feat['layer4'], x, feat_masks, None, i, GX, True])
            rw_bias = [rw2, rw3, rw4, rw5]
            rw_scale = [rw2_g, rw3_g, rw4_g, rw5_g]

            c2, m2 = self.mix2([x, rw_bias, rw_scale])
            x = x * m2 + c2

            x = self.layer2(x)

            c3, m3 = self.mix3([x, rw_bias, rw_scale])
            x = x * m3 + c3

            x = self.layer3(x)

            c4, m4 = self.mix4([x, rw_bias, rw_scale])
            x = x * m4 + c4

            x = self.layer4(x)

            c5, m5 = self.mix5([x, rw_bias, rw_scale])
            x = x * m5 + c5

        else:
            rw2, rw2_g = self.rw2([feat['layer1'], x, feat_masks, None, i, GX, False])
            x = x * rw2_g + rw2

            x = self.layer2(x)

            rw3, rw3_g = self.rw3([feat['layer2'], x, feat_masks, None, i, GX, False])
            x = x * rw3_g + rw3

            x = self.layer3(x)

            rw4, rw4_g = self.rw4([feat['layer3'], x, feat_masks, None, i, GX, False])
            x = x * rw4_g + rw4

            x = self.layer4(x)

            rw5, rw5_g = self.rw5([feat['layer4'], x, feat_masks, None, i, GX, False])
            x = x * rw5_g + rw5

        x = self.bn5(self.conv5(x))
        x = self.relu(x)
        x = self.avgpool(x)
        if self.extract_feat:
            return x

        x = self.fc1(x)
        x = self.relu(x)

        if self.task == 'class':
            x = x.squeeze(4).squeeze(3).squeeze(2) # B C
            x = self.dropout(x)
            x = self.fc2(x).unsqueeze(2) # B C 1
        if self.task == 'loc':
            x = x.squeeze(4).squeeze(3).permute(0,2,1) # B T C
            x = self.dropout(x)
            x = self.fc2(x).permute(0,2,1) # B C T


        x = x.unsqueeze(3).unsqueeze(4)
        rw6, rw6_g = self.rw6([feat['conv5'], x, feat_masks, None, i, GX, False])
        x = (x * rw6_g + rw6).squeeze(4).squeeze(3)

        if self.t_pool == 'grid':
            x = GridUnpool([x, gx, True])
            x = F.interpolate(x, (x.shape[2]-1)*4, mode='linear', align_corners=True)

        return x


def replace_logits(self, n_classes):
        self.fc2 = nn.Linear(2048, n_classes)

def get_inplanes(version):
    planes = {'S':[(54,24), (108,48), (216,96), (432,192)],
              'M':[(54,24), (108,48), (216,96), (432,192)],
              'XL':[(72,32), (162,72), (306,136), (630,280)]}

    return planes[version]

def get_blocks(version):
    blocks = {'S':[3,5,11,7],
              'M':[3,5,11,7],
              'XL':[5,10,25,15]}

    return blocks[version]

def generate_model(x3d_version, **kwargs):
    model = ResNet(Bottleneck, get_blocks(x3d_version), get_inplanes(x3d_version), **kwargs)

    return model
