# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import BACKBONES
from ..builder import HEADS
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmseg.ops import resize


class QKVBlock(nn.Module):
    def __init__(self, in_channels):
        super(QKVBlock, self).__init__()
        self.conv_q = ConvModule(in_channels, in_channels, 1, norm_cfg=None, act_cfg=None)
        self.conv_k = ConvModule(in_channels, in_channels, 1, norm_cfg=None, act_cfg=None)
        self.conv_v = ConvModule(in_channels, in_channels, 1, norm_cfg=None, act_cfg=None)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, C, H, W = x.shape

        x_q = self.conv_q(x).reshape(N, C, H*W).permute(0, 2, 1).contiguous()
        x_k = self.conv_k(x).reshape(N, C, H*W)
        x_v = self.conv_v(x).reshape(N, C, H*W).permute(0, 2, 1).contiguous()

        att = self.softmax(torch.bmm(x_q, x_k))

        return x_v, att


class MCAM(nn.Module):
    def __init__(self, in_channels):
        super(MCAM, self).__init__()
        self.qkv_opt = QKVBlock(in_channels)
        self.qkv_sar = QKVBlock(in_channels)

    def forward(self, x_opt, x_sar):

        N, C, H, W = x_opt.shape

        v_opt, att_opt = self.qkv_opt(x_opt)
        v_sar, att_sar = self.qkv_sar(x_sar)
        att = att_opt * att_sar

        s_opt = torch.bmm(att, v_opt).permute(0, 2, 1).contiguous().reshape(N, C, H, W)
        s_sar = torch.bmm(att, v_sar).permute(0, 2, 1).contiguous().reshape(N, C, H, W)

        out = s_opt * s_sar
        return out


'''
Copied from mmseg.models.decode_heads.aspp_head
'''
class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


class ASPP(nn.Module):
    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        super(ASPP, self).__init__()
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels,
                channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.aspp_modules = ASPPModule(dilations, in_channels, channels, conv_cfg, norm_cfg, act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * channels,
            channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""

        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=False)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        return output


@BACKBONES.register_module()
class MCANet(BaseModule):
    def __init__(self,
                 enc_dims=[256, 2048],
                 backbone_opt_cfg=dict(
                     type='ResNetV1c',
                     depth=50,
                     in_channels=4,
                     num_stages=4,
                     out_indices=(0, 3),
                     dilations=(1, 1, 2, 4),
                     strides=(1, 2, 1, 1),
                     norm_cfg=dict(type='BN', requires_grad=True),
                     norm_eval=False,
                     style='pytorch',
                     contract_dilation=True,
                     pretrained='open-mmlab://resnet50_v1c'
                 ),
                 backbone_sar_cfg=dict(
                     type='ResNetV1c',
                     depth=50,
                     in_channels=1,
                     num_stages=4,
                     out_indices=(0, 3),
                     dilations=(1, 1, 2, 4),
                     strides=(1, 2, 1, 1),
                     norm_cfg=dict(type='BN', requires_grad=True),
                     norm_eval=False,
                     style='pytorch',
                     contract_dilation=True,
                     pretrained='open-mmlab://resnet50_v1c'
                 ),
                 norm_cfg=dict(type='BN')):
        super(MCANet, self).__init__()

        self.backbone_opt = BACKBONES.build(backbone_opt_cfg)
        self.backbone_sar = BACKBONES.build(backbone_sar_cfg)

        self.mcam_low = MCAM(enc_dims[0])
        self.conv_fuse_low = ConvModule(enc_dims[0]*3, 48, 1, norm_cfg=norm_cfg)

        self.mcam_high = MCAM(enc_dims[1])
        self.conv_opt_high = ConvModule(enc_dims[1], 256, 1, norm_cfg=norm_cfg)
        self.conv_sar_high = ConvModule(enc_dims[1], 256, 1, norm_cfg=norm_cfg)
        self.aspp = ASPP(dilations=(1, 6, 12, 18),
                         in_channels=enc_dims[1]+2*256,
                         channels=256,
                         conv_cfg=None,
                         norm_cfg=norm_cfg,
                         act_cfg=dict(type='ReLU'))


    def forward(self, x):
        # fetch inputs
        x_opt, x_sar = torch.split(x, (4, 1), dim=1)

        # pseudo-siamese feature extraction
        x_opt_low, x_opt_high = self.backbone_opt(x_opt)
        x_sar_low, x_sar_high = self.backbone_sar(x_sar)

        # multimodal-cross attention
        x_fuse_low = self.mcam_low(x_opt_low, x_sar_low)
        x_fuse_low = torch.cat([x_fuse_low, x_opt_low, x_sar_low], dim=1)
        x_fuse_low = self.conv_fuse_low(x_fuse_low)
        low_h, low_w = x_fuse_low.size(2), x_fuse_low.size(3)

        x_fuse_high = self.mcam_high(x_opt_high, x_sar_high)
        x_fuse_high = torch.cat([x_fuse_high, self.conv_opt_high(x_opt_high), self.conv_sar_high(x_sar_high)], dim=1)
        x_fuse_high = self.aspp(x_fuse_high)
        x_fuse_high = F.interpolate(x_fuse_high, size=(int(low_h), int(low_w)), mode='bilinear')
        # decoder
        x_merge = torch.cat([x_fuse_high, x_fuse_low], dim=1)
        return (x_merge, )
