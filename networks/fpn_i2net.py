import torch
import torch.nn as nn
from torch.nn import functional as F
from .base import ASPP,PPM
from .i2net import i2net_simfpn
import math
import numpy as np

def get_syncbn():
    #return nn.BatchNorm2d
    return nn.SyncBatchNorm


class fpn_i2net(nn.Module):

    def __init__(self, in_planes, num_classes=19, inner_planes=256, sync_bn=False, dilations=(12, 24, 36), 
                pos_dim=24, ultra_pe=True, unfold=False, no_aspp=True,no_ppm=True,
                local=False, stride=1, learn_pe=False, require_grad=True, num_layer=2,
                K=4, gate_layer=1, *args):

        super(fpn_i2net, self).__init__()
        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        
        assert not(no_aspp==False and no_ppm==False)

        self.no_aspp = no_aspp
        self.no_ppm=no_ppm

        self.unfold = unfold
      
        
        if self.no_aspp and self.no_ppm:
            self.head = nn.Sequential(nn.Conv2d(in_planes[-1], inner_planes, kernel_size=1), norm_layer(inner_planes), nn.ReLU(inplace=True))
        elif not self.no_aspp:
            self.aspp = ASPP(in_planes[-1], inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)
            self.head = nn.Sequential(
                nn.Conv2d(self.aspp.get_outplanes(), 256, kernel_size=3, padding=1, dilation=1, bias=False),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1))
        elif not self.no_ppm:
            self.ppm = PPM(in_planes[-1], inner_planes=inner_planes, sync_bn=sync_bn)
            self.head = nn.Sequential(
                nn.Conv2d(self.ppm.get_outplanes(), 256, kernel_size=3, padding=1, dilation=1, bias=False),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1))
        
        self.ifa = i2net_simfpn(plane=inner_planes,ultra_pe=ultra_pe, pos_dim=pos_dim,sync_bn=sync_bn, 
                                num_classes=num_classes, local=local, unfold=unfold, stride=stride, 
                                learn_pe=learn_pe, require_grad=require_grad, num_layer=num_layer,
                                K=K,gate_layer=gate_layer)
        self.enc1 = nn.Sequential(nn.Conv2d(in_planes[0], inner_planes, kernel_size=1), norm_layer(inner_planes), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(in_planes[1], inner_planes, kernel_size=1), norm_layer(inner_planes), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(in_planes[2], inner_planes, kernel_size=1), norm_layer(inner_planes), nn.ReLU(inplace=True))
    
    def forward(self, x,return_gate=False,**kwargs):
        x1, x2, x3, x4 = x
        if not self.no_aspp:
            x4 = self.aspp(x4)
            x4 = self.head(x4)
        elif not self.no_ppm:
            x4 = self.ppm(x4)
            x4 = self.head(x4)
        else:
            x4 = self.head(x4)

        x1 = self.enc1(x1) # bsz*C*h*w
        x2 = self.enc2(x2)
        x3 = self.enc3(x3)
        context = []
        h, w = x1.shape[-2], x1.shape[-1]


        target_feat = [x1, x2, x3, x4]

        for i, feat in enumerate(target_feat):
            context.append(self.ifa(feat, size=[h, w], level=i+1)) # upsample to h*w maps by finding the nearest basis on the given level map
        context = torch.cat(context, dim=-1).permute(0,2,1) # (bsz,C,h*w)

        if return_gate:
            res,gate = self.ifa(context, size=[h, w], after_cat=True, return_gate=True) # (bsz,cls,h,w)
            return res,gate
        else:
            res = self.ifa(context, size=[h, w], after_cat=True) # (bsz,cls,h,w)
            return res
