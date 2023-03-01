import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from ..builder import BACKBONES
from mmcv.cnn import build_norm_layer
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import BaseModule


class RCB(nn.Module):

    def __init__(self, features, out_features=256, norm_cfg=dict(type='BN', requires_grad=True)):
        super(RCB, self).__init__()

        self.unify = nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False)
        self.residual = nn.Sequential(
            nn.Conv2d(out_features, out_features // 4, kernel_size=3, padding=1, dilation=1, bias=False),
            build_norm_layer(norm_cfg, out_features // 4)[1],
            nn.ReLU(),
            nn.Conv2d(out_features // 4, out_features, kernel_size=3, padding=1, dilation=1, bias=False))
        self.norm = build_norm_layer(norm_cfg, out_features)[1]
        self.act = nn.ReLU()

    def forward(self, feats):
        feats = self.unify(feats)
        residual = self.residual(feats)
        feats = self.act(self.norm(feats + residual))
        return feats


class CAB(nn.Module):
    def __init__(self, features, norm_cfg=dict(type='BN', requires_grad=True)):
        super(CAB, self).__init__()

        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, features)[1],
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, features)[1],
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen1[2].weight.data.zero_()
        self.delta_gen2[2].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[w / s, h / s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)

        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)

        high_stage += low_stage
        return high_stage


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        dilate1_out = self.act(self.dilate1(x))
        dilate2_out = self.act(self.dilate2(dilate1_out))
        dilate3_out = self.act(self.dilate3(dilate2_out))
        dilate4_out = self.act(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DSC(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1):
        super(DSC, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Upsample(nn.Module):

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels):
        super(DecoderBlock, self).__init__()

        self.identity = nn.Sequential(
            Upsample(2, mode="bilinear"),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        )

        self.decode = nn.Sequential(
            Upsample(2, mode="bilinear"),
            nn.BatchNorm2d(input_channels),
            DSC(input_channels, input_channels),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            DSC(input_channels, output_channels),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        residual = self.identity(x)

        out = self.decode(x)

        out += residual

        return out


class GatedFusion(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        G = self.gate(out)

        PG = x * G
        FG = y * (1 - G)

        return FG + PG


class SpatialQKVBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(SpatialQKVBlock, self).__init__()
        self.conv_q = ConvModule(in_channels, channels, 1, norm_cfg=None, act_cfg=None)
        self.conv_k = ConvModule(in_channels, channels, 1, norm_cfg=None, act_cfg=None)
        self.conv_v = ConvModule(in_channels, channels, 1, norm_cfg=None, act_cfg=None)

    def forward(self, x):
        N, C, H, W = x.shape

        x_q = self.conv_q(x).reshape(N, -1, H*W).permute(0, 2, 1).contiguous()
        x_k = self.conv_k(x).reshape(N, -1, H*W)
        x_v = self.conv_v(x).reshape(N, -1, H*W).permute(0, 2, 1).contiguous()

        return x_q, x_k, x_v


class SpatialAttBlock(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(SpatialAttBlock, self).__init__()
        self.qkv_opt = SpatialQKVBlock(in_channels, channels)
        self.qkv_sar = SpatialQKVBlock(in_channels, channels)
        self.gamma_opt = Scale(0)
        self.gamma_sar = Scale(0)
        self.project_opt = ConvModule(channels, out_channels, kernel_size=1, norm_cfg=None, act_cfg=None)
        self.project_sar = ConvModule(channels, out_channels, kernel_size=1, norm_cfg=None, act_cfg=None)

    def forward(self, x_opt, x_sar):
        N, C, H, W = x_opt.shape

        q_opt, k_opt, v_opt = self.qkv_opt(x_opt)
        q_sar, k_sar, v_sar = self.qkv_sar(x_sar)

        att_opt = F.softmax(torch.bmm(q_opt, k_opt), dim=-1)
        att_sar = F.softmax(torch.bmm(q_sar, k_sar), dim=-1)
        att = torch.bmm(att_opt, att_sar)

        s_opt = torch.bmm(att, v_opt).permute(0, 2, 1).contiguous().reshape(N, -1, H, W)
        s_opt = self.project_opt(s_opt)
        s_opt = x_opt + self.gamma_opt(s_opt)

        s_sar = torch.bmm(att, v_sar).permute(0, 2, 1).contiguous().reshape(N, -1, H, W)
        s_sar = self.project_sar(s_sar)
        s_sar = x_sar + self.gamma_sar(s_sar)

        return s_opt + s_sar


class ChannelQKVBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(ChannelQKVBlock, self).__init__()
        self.conv_q = ConvModule(in_channels, channels, 1, norm_cfg=None, act_cfg=None)
        self.conv_k = ConvModule(in_channels, channels, 1, norm_cfg=None, act_cfg=None)
        self.conv_v = ConvModule(in_channels, channels, 1, norm_cfg=None, act_cfg=None)

    def forward(self, x):
        N, C, H, W = x.shape

        x_q = self.conv_q(x).reshape(N, -1, H*W)
        x_k = self.conv_k(x).reshape(N, -1, H*W).permute(0, 2, 1).contiguous()
        x_v = self.conv_v(x).reshape(N, -1, H*W)

        return x_q, x_k, x_v


class ChannelAttBlock(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(ChannelAttBlock, self).__init__()
        self.qkv_opt = ChannelQKVBlock(in_channels, channels)
        self.qkv_sar = ChannelQKVBlock(in_channels, channels)
        self.gamma_opt = Scale(0)
        self.gamma_sar = Scale(0)
        self.project_opt = ConvModule(channels, out_channels, kernel_size=1, norm_cfg=None, act_cfg=None)
        self.project_sar = ConvModule(channels, out_channels, kernel_size=1, norm_cfg=None, act_cfg=None)

    def forward(self, x_opt, x_sar):
        N, C, H, W = x_opt.shape

        q_opt, k_opt, v_opt = self.qkv_opt(x_opt)
        q_sar, k_sar, v_sar = self.qkv_sar(x_sar)

        att_opt = F.softmax(torch.bmm(q_opt, k_opt), dim=-1)
        att_sar = F.softmax(torch.bmm(q_sar, k_sar), dim=-1)

        att = torch.bmm(att_opt, att_sar)

        s_opt = torch.bmm(att, v_opt).reshape(N, -1, H, W)
        s_opt = self.project_opt(s_opt)
        s_opt = x_opt + self.gamma_opt(s_opt)

        s_sar = torch.bmm(att, v_sar).reshape(N, -1, H, W)
        s_sar = self.project_sar(s_sar)
        s_sar = x_sar + self.gamma_sar(s_sar)

        return s_opt + s_sar


class AttFuseBlock(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(AttFuseBlock, self).__init__()
        self.sab = SpatialAttBlock(in_channels, channels, out_channels)
        self.cab = ChannelAttBlock(in_channels, channels, out_channels)

    def forward(self, x_opt, x_sar):
        sab_feat = self.sab(x_opt, x_sar)
        cab_feat = self.cab(x_opt, x_sar)
        return sab_feat + cab_feat


class AlignDecoderBlock(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(AlignDecoderBlock, self).__init__()

        self.up = CAB(input_channels, norm_cfg)
        self.identity_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        self.decode = nn.Sequential(
            build_norm_layer(norm_cfg, input_channels)[1],
            DSC(input_channels, input_channels),
            build_norm_layer(norm_cfg, input_channels)[1],
            nn.ReLU(inplace=True),
            DSC(input_channels, output_channels),
            build_norm_layer(norm_cfg, output_channels)[1],
        )

    def forward(self, low_feat, high_feat):
        f = self.up(low_feat, high_feat)
        residual = self.identity_conv(f)
        out = self.decode(f)
        out += residual
        return out


@BACKBONES.register_module()
class CMCD(BaseModule):
    def __init__(self,
                 enc_opt_dims=[64, 256, 512, 1024, 2048],
                 backbone_opt_cfg=dict(type='TIMMBackbone',
                                       model_name='resnet50',
                                       in_channels=4,
                                       out_indices=(0, 1, 2, 3, 4),
                                       output_stride=32,
                                       pretrained=True),
                 enc_sar_dims=[64, 64, 128, 256, 512],
                 backbone_sar_cfg=dict(type='TIMMBackbone',
                                       model_name='resnet18',
                                       in_channels=1,
                                       out_indices=(0, 1, 2, 3, 4),
                                       output_stride=32,
                                       pretrained=True),
                 center_block="dblock",
                 side_dim=64,
                 att_dim_factor=2,
                 norm_cfg=dict(type='BN')
                 ):
        super(CMCD, self).__init__()

        # RGB feature extractor
        self.backbone_opt = BACKBONES.build(backbone_opt_cfg)

        # SAR feature extractor
        self.backbone_sar = BACKBONES.build(backbone_sar_cfg)

        # bridge module
        if center_block == 'dblock':
            self.center_opt = Dblock(enc_opt_dims[-1])
            self.center_sar = Dblock(enc_sar_dims[-1])
        else:
            self.center_opt = nn.Identity()
            self.center_sar = nn.Identity()

        # side features
        self.side1_rgb = RCB(enc_opt_dims[0], side_dim, norm_cfg=norm_cfg)
        self.side2_rgb = RCB(enc_opt_dims[1], side_dim, norm_cfg=norm_cfg)
        self.side3_rgb = RCB(enc_opt_dims[2], side_dim, norm_cfg=norm_cfg)
        self.side4_rgb = RCB(enc_opt_dims[3], side_dim, norm_cfg=norm_cfg)
        self.side5_rgb = RCB(enc_opt_dims[4], side_dim, norm_cfg=norm_cfg)

        self.side1_sar = RCB(enc_sar_dims[0], side_dim, norm_cfg=norm_cfg)
        self.side2_sar = RCB(enc_sar_dims[1], side_dim, norm_cfg=norm_cfg)
        self.side3_sar = RCB(enc_sar_dims[2], side_dim, norm_cfg=norm_cfg)
        self.side4_sar = RCB(enc_sar_dims[3], side_dim, norm_cfg=norm_cfg)
        self.side5_sar = RCB(enc_sar_dims[4], side_dim, norm_cfg=norm_cfg)

        # cross modal fusion
        self.fuse1 = GatedFusion(side_dim)
        self.fuse2 = GatedFusion(side_dim)
        self.fuse3 = GatedFusion(side_dim)
        self.fuse4 = GatedFusion(side_dim)
        self.fuse5 = AttFuseBlock(side_dim, side_dim // att_dim_factor, side_dim)
        self.final_fuse = GatedFusion(side_dim)

        # RGB decoders
        self.decode1_rgb = DecoderBlock(side_dim, side_dim)
        self.decode2_rgb = AlignDecoderBlock(side_dim, side_dim, norm_cfg=norm_cfg)
        self.decode3_rgb = AlignDecoderBlock(side_dim, side_dim, norm_cfg=norm_cfg)
        self.decode4_rgb = AlignDecoderBlock(side_dim, side_dim, norm_cfg=norm_cfg)
        self.decode5_rgb = AlignDecoderBlock(side_dim, side_dim, norm_cfg=norm_cfg)

        # SAR decoders
        self.decode1_sar = DecoderBlock(side_dim, side_dim)
        self.decode2_sar = AlignDecoderBlock(side_dim, side_dim, norm_cfg=norm_cfg)
        self.decode3_sar = AlignDecoderBlock(side_dim, side_dim, norm_cfg=norm_cfg)
        self.decode4_sar = AlignDecoderBlock(side_dim, side_dim, norm_cfg=norm_cfg)
        self.decode5_sar = AlignDecoderBlock(side_dim, side_dim, norm_cfg=norm_cfg)

    def forward(self, x):

        # fetch inputs
        x_img, x_aux = torch.split(x, (4, 1), dim=1)

        # encode x
        x1, x2, x3, x4, x5 = self.backbone_opt(x_img)
        x5 = self.center_opt(x5)
        x1_side = self.side1_rgb(x1)
        x2_side = self.side2_rgb(x2)
        x3_side = self.side3_rgb(x3)
        x4_side = self.side4_rgb(x4)

        x5_side = self.side5_rgb(x5)

        # encode y
        y1, y2, y3, y4, y5 = self.backbone_sar(x_aux)
        y5 = self.center_sar(y5)
        y1_side = self.side1_sar(y1)
        y2_side = self.side2_sar(y2)
        y3_side = self.side3_sar(y3)
        y4_side = self.side4_sar(y4)
        y5_side = self.side5_sar(y5)

        # side fuse
        y5_side = self.fuse5(x5_side, y5_side)
        y4_side = self.fuse4(x4_side, y4_side)
        y3_side = self.fuse3(x3_side, y3_side)
        y2_side = self.fuse2(x2_side, y2_side)
        y1_side = self.fuse1(x1_side, y1_side)

        # decode stage 5
        out_rgb = self.decode5_rgb(x4_side, x5_side)

        out_sar = self.decode5_sar(y4_side, y5_side)

        # decode stage 4
        out_rgb = self.decode4_rgb(x3_side, out_rgb)

        out_sar = self.decode4_sar(y3_side, out_sar)

        # decode stage 3
        out_rgb = self.decode3_rgb(x2_side, out_rgb)

        out_sar = self.decode3_sar(y2_side, out_sar)

        # decode stage 2
        out_rgb = self.decode2_rgb(x1_side, out_rgb)

        out_sar = self.decode2_sar(y1_side, out_sar)

        # decode stage 1
        out_rgb = self.decode1_rgb(out_rgb)

        out_sar = self.decode1_sar(out_sar)

        # final fuse
        f_final = self.final_fuse(out_rgb, out_sar)

        return (f_final,)