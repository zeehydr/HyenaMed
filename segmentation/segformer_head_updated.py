# Copyright (c) OpenMMLab. All rights reserved.
# Please copy this head and past in `mmsegmentation\mmseg\models\decode_heads`
# Second step register this new head in the `__init__.py` file
# at the top add this line: `from .segformer_head_updated import SegformerHeadUpdated`
# also add the new head to `__all__ = [ ..., 'SegformerHeadUpdated']

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', 
                              groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', 
                              groups=dim, dilation=dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x



class AxialDWWithPooling(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1, reduction=16):    
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', 
                              groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', 
                              groups=dim, dilation=dilation)
        
        # Channel Mixing
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x + self.dw_h(x) + self.dw_w(x)
        
        # Channel Mixing
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        pooling = self.sigmoid(avg_out + max_out)
        
        return identity + x * pooling

@MODELS.register_module()
class SegformerHeadUpdated(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
        use_axial_dw: Whether to use AxialDW block. Default: False.
        axial_kernel: Kernel size for AxialDW. Default: (7, 7).
        axial_dilation: Dilation for AxialDW. Default: 1.
    """

    def __init__(self, interpolate_mode='bilinear', use_axial_dw=True, use_pooling=True,
                 axial_kernel=(7, 7), axial_dilation=1, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        self.use_axial_dw = use_axial_dw
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        if self.use_axial_dw:
            # First pointwise convolution
            self.pw = nn.Conv2d(self.channels, self.channels, kernel_size=1)
            # BatchNorm
            self.bn = nn.BatchNorm2d(self.channels)
            # ReLU activation
            self.act = nn.ReLU()
            # self.act = nn.GELU()
            # Second pointwise convolution
            self.pw2 = nn.Conv2d(self.channels, self.channels, kernel_size=1)
            # Choose between regular AxialDW or with Pooling
            if use_pooling:
                self.adw = AxialDWWithPooling(self.channels, mixer_kernel=axial_kernel, 
                                                dilation=axial_dilation)
            else:
                self.adw = AxialDW(self.channels, mixer_kernel=axial_kernel, 
                                   dilation=axial_dilation)        

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        identity = out # Save for residual connection in V2
        
        # Apply the new components if use_axial_dw is True
        if self.use_axial_dw:
            out = self.pw(out)
            out = self.bn(out)
            out = self.act(out)
            out = self.pw2(out)
            # out = self.adw(out)

            # Add skip connection (residual) in V2
            out = out + identity

        out = self.cls_seg(out)

        return out
