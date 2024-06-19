import torch
from torch.functional import unique
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import constant_

Interpolate=F.interpolate


def get_syncbn():
    #return nn.BatchNorm2d
    return nn.SyncBatchNorm

def get_bn():
    return nn.BatchNorm


def conv_transpose(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False):
    layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1, dilation=dilation, bias=bias)
    # Bilinear interpolation init
    w = torch.Tensor(kernel_size, kernel_size)
    centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
    for y in range(kernel_size):
        for x in range(kernel_size):
            w[y, x] = (1 - abs((x - centre) / stride)) * (1 - abs((y - centre) / stride))
    layer.weight.data.copy_(w.div(in_planes).repeat(in_planes, out_planes, 1, 1))
    if bias:
        constant_(layer.bias, 0)
    return layer


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)

    
    
class PPM(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_planes, inner_planes=256, sizes=(1, 2, 3, 6), sync_bn=False):
        super(PPM, self).__init__()
        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_planes, inner_planes, size, norm_layer) for size in sizes])
        
        self.out_planes=in_planes+len(sizes)*inner_planes

    def _make_stage(self, in_planes, inner_planes, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_planes, inner_planes, kernel_size=1, bias=False)
        bn = norm_layer(inner_planes)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        bottle = torch.cat(priors, 1)
        return bottle
    
    def get_outplanes(self):
        return self.out_planes



class ASPP(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, in_planes, inner_planes=256, sync_bn=False, dilations=(12, 24, 36)):
        super(ASPP, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                   # norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                   padding=dilations[0], dilation=dilations[0], bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                   padding=dilations[1], dilation=dilations[1], bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                   padding=dilations[2], dilation=dilations[2], bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))

        self.out_planes = (len(dilations) + 2) * inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=False)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return aspp_out

