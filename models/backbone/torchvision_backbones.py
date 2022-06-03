"""
Backbones supported by torchvison.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.models import resnet101
from fastai.vision.models.unet import DynamicUnet
import torchvision

class TVDeeplabRes101Encoder(nn.Module):
    """
    FCN-Resnet101 backbone from torchvision deeplabv3
    No ASPP is used as we found emperically it hurts performance
    """
    def __init__(self, shared_encoder ,use_coco_init, aux_dim_keep = 64, use_aspp = False):
        super().__init__()
        
        self.backbone = shared_encoder
        self.localconv = nn.Conv2d(2048, 256,kernel_size = 1, stride = 1, bias = False) # reduce feature map dimension
        self.asppconv = nn.Conv2d(256, 256,kernel_size = 1, bias = False)

        self.use_aspp = use_aspp

    def forward(self, x_in, low_level):
        """
        Args:
            low_level: whether returning aggregated low-level features in FCN
        """
        fts = self.backbone(x_in)
        if self.use_aspp:
            fts256 = self.aspp_out(fts['out'])
            high_level_fts = fts256
        else:
            fts2048 = fts
            high_level_fts = self.localconv(fts2048)

        if low_level:
            low_level_fts = fts['aux'][:, : self.aux_dim_keep]
            return high_level_fts, low_level_fts
        else:
            return high_level_fts