""" Full assembly of the parts to form the complete network """
#https://github.com/milesial/Pytorch-UNet

import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from .unet_utils import *
from .fpn_i2net import fpn_i2net


class GlasUNetI2Net(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.encoder = get_encoder(
            config.encoder_model,
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        # out_channels=[3,64,64,128,256,512]
        # in_planes=[64, 128, 256, 512]
        self.decoder=fpn_i2net(in_planes=self.encoder.out_channels[-4:],inner_planes=config.inner_planes,
                               pos_dim=config.pos_dim,ultra_pe=config.ultra_pe,no_aspp=config.no_aspp,
                               num_layer=config.num_layer,num_classes=64,
                               K=config.K,gate_layer=config.gate_layer)
        
        self.outc = (OutConv(64+64, config.n_class))
        
    def forward(self, x, return_gate=False):
        features = self.encoder(x) # [3,64,64,128,256,512]
        y= [features[-4+i] for i in range(4)]
        if return_gate:
            y,gate = self.decoder(y,return_gate) # [bsz,64,128,128]
            y = torch.cat([features[1],F.interpolate(y,size=features[1].shape[-2:],mode='bilinear')],dim=1) # [bsz,64+64,256,256]
            logits = self.outc(y)
            logits = logits if logits.shape[-1]==512 else F.interpolate(logits,size=[512,512],mode='bilinear')
            return logits,gate
        else:
            y = self.decoder(y) # [bsz,64,128,128]
            y = torch.cat([features[1],F.interpolate(y,size=features[1].shape[-2:],mode='bilinear')],dim=1) # [bsz,64+64,256,256]
            logits = self.outc(y)

            return logits if logits.shape[-1]==512 else F.interpolate(logits,size=[512,512],mode='bilinear')