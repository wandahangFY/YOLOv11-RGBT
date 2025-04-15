import math

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from timm.models.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, create_attn, get_attn, create_classifier
from functools import partial
import torch.nn.functional as F
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad

__all__ = (
    "CSPResNet_CBS","CSPResNet","ConvBNLayer","ResSPP","CoordConv",
    "ResNet50vd","ResNet50vd_dcn","ResNet101vd","PPConvBlock","Res2net50"
)

# PPYOLO系列的原始文件来自于
# https://github.com/iscyy/yoloair
# https://github.com/Nioolek/PPYOLOE_pytorch
# https://github.com/PaddlePaddle/PaddleDetection

#PPYOLOE-L


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self, idx,inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)
        self.idx = idx
        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)

        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self, idx,net_block_idx,bool_DeformableConv2d,inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()
        self.idx = idx
        self.bool_DeformableConv2d = bool_DeformableConv2d
        self.net_block_idx = net_block_idx
        last_layer = layers[0]+layers[1]+layers[2]
        self.last_layer = last_layer
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

        if self.net_block_idx==last_layer:
            if bool_DeformableConv2d:
                self.dcn_v2_down = DeformableConv2d(first_planes, first_planes, 3, 2)
        if self.net_block_idx > last_layer:
            if bool_DeformableConv2d:
                self.dcn_v2 = DeformableConv2d(first_planes,first_planes,3,1)



    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):

        self.idxx = self.idx +1
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.net_block_idx==self.last_layer:
            if self.bool_DeformableConv2d:
                x = self.dcn_v2_down(x)
            else:
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.drop_block(x)
                x = self.act2(x)
                x = self.aa(x)
        elif self.net_block_idx>self.last_layer:
            if self.bool_DeformableConv2d:
                x = self.dcn_v2(x)
            else:
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.drop_block(x)
                x = self.act2(x)
                x = self.aa(x)
        else:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.drop_block(x)
            x = self.act2(x)
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x

class Bottle2neck(nn.Module):
    """ Res2Net/Res2NeXT Bottleneck
    Adapted from https://github.com/gasvn/Res2Net/blob/master/res2net.py
    """
    expansion = 4

    def __init__(
            self, idx,net_block_idx,bool_DeformableConv2d,inplanes, planes, stride=1, downsample=None,
            cardinality=1, base_width=26, scale=4, dilation=1, first_dilation=None,
            act_layer=nn.ReLU, norm_layer=None, attn_layer=None, **_):
        super(Bottle2neck, self).__init__()
        self.scale = scale
        self.is_first = stride > 1 or downsample is not None
        self.num_scales = max(1, scale - 1)
        width = int(math.floor(planes * (base_width / 64.0))) * cardinality
        self.width = width
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width * scale)

        convs = []
        bns = []
        for i in range(self.num_scales):
            convs.append(nn.Conv2d(
                width, width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, bias=False))
            bns.append(norm_layer(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        if self.is_first:
            # FIXME this should probably have count_include_pad=False, but hurts original weights
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            self.pool = None

        self.conv3 = nn.Conv2d(width * scale, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.se = attn_layer(outplanes) if attn_layer is not None else None

        self.relu = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        spo = []
        sp = spx[0]  # redundant, for torchscript
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0 or self.is_first:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)
        if self.scale > 1:
            if self.pool is not None:  # self.is_first == True, None check for torchscript
                spo.append(self.pool(spx[-1]))
            else:
                spo.append(spx[-1])
        out = torch.cat(spo, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)

        return out



class Res2net50(nn.Module):
    """Constructs a Res2Net-50  model."""

    def __init__(self,cout = 64,idx = 0):

        super(Res2net50 , self).__init__()
        self.cout = cout
        self.idx = idx
        self.res2net50  = ResNet(cout = cout,idx = idx,channels = [64, 128, 256, 512] ,block=Bottle2neck, layers=[3, 4, 6, 3], cardinality=8,base_width=4)
    def forward(self, x):
        x = self.res2net50(x)
        return x

class ResSPP(nn.Module):   #res SPP

    def __init__(self, c1 = 1024 ,c2 = 384,n = 3,act='swish',k = (5,9,13)):
        super(ResSPP, self).__init__()
        c_ = c2
        if c2 == 1024:
            c_ = c2//2
        self.conv1 = ConvBNLayer(c1, c_, 1, act=act)  # CBR
        self.basicBlock_spp1 = BasicBlock(c_, c_,shortcut=False)
        self.basicBlock_spp2 = BasicBlock(c_, c_, shortcut=False)
        self.spp =nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.conv2 = ConvBNLayer(c_*4, c_, 1, act=act)
        self.basicBlock_spp3 = BasicBlock(c_, c_, shortcut=False)
        self.basicBlock_spp4 = BasicBlock(c_, c_, shortcut=False)
        self.n = n



    def forward(self, x):
        y1 = self.conv1(x)
        if self.n == 3:
            y1 = self.basicBlock_spp1(y1)
            y1 = self.basicBlock_spp2(y1)
            y1 = torch.cat([y1] + [m(y1) for m in self.spp], 1)
            y1 = self.conv2(y1)
            y1 = self.basicBlock_spp3(y1)
        elif self.n == 1:
            y1 = self.basicBlock_spp1(y1)
            y1 = torch.cat([y1] + [m(y1) for m in self.spp], 1)
            y1 = self.conv2(y1)
        elif self.n == 2:
            y1 = self.basicBlock_spp1(y1)
            y1 = torch.cat([y1] + [m(y1) for m in self.spp], 1)
            y1 = self.conv2(y1)
            y1 = self.basicBlock_spp2(y1)
        elif self.n == 4:
            y1 = self.basicBlock_spp1(y1)
            y1 = self.basicBlock_spp2(y1)
            y1 = torch.cat([y1] + [m(y1) for m in self.spp], 1)
            y1 = self.conv2(y1)
            y1 = self.basicBlock_spp3(y1)
            y1 = self.basicBlock_spp4(y1)
        return y1

#https://github.com/Nioolek/PPYOLOE_pytorch
#https://github.com/PaddlePaddle/PaddleDetection
class CSPResNet(nn.Module):

    def __init__(self,c1,c2,n,conv_down,infor = 'backbone',act='swish'):
        super(CSPResNet, self).__init__()
        self.backbone = CSPResStage(BasicBlock, c1, c2, n,conv_down,infor, act=act)

    def forward(self, x):
        x = self.backbone(x)
        return x


class CSPResNet_CBS(nn.Module):

    def __init__(self,c1=3,c2=64,use_large_stem=True,act='swish'):
        super(CSPResNet_CBS, self).__init__()
        if use_large_stem:
            self.stem = nn.Sequential(
                (ConvBNLayer(c1, c2 // 2, 3, stride=2, padding=1,act = act)),
                (ConvBNLayer(c2 // 2,c2 // 2,3,stride=1,padding=1,act = act)),
                (ConvBNLayer(c2 // 2,c2,3,stride=1,padding=1,act = act))
            )
        else:
            self.stem = nn.Sequential(
                (ConvBNLayer(3, c2 // 2, 3, stride=2, padding=1,act = act)),
                (ConvBNLayer(c2 // 2,c2,3,stride=1,padding=1,act = act)))

    def forward(self, x):
        x = self.stem(x)
        return x

class ConvBNLayer(nn.Module):  #CBS,CBR

    def __init__(self, ch_in, ch_out, filter_size=3, stride=1, groups=1, padding=0, act='swish'):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(ch_out, )
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

#CSPRes
class CSPResStage(nn.Module):
    def __init__(self, block_fn, c1, c2, n, stride, infor = "backbone",act='relu', attn='eca'):
        super(CSPResStage, self).__init__()
        ch_mid = (c1 + c2) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(c1, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)  #CBR 1x1,BN,RELU
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)  # CBR 1x1,BN,RELU
        self.blocks = nn.Sequential(*[block_fn(ch_mid // 2, ch_mid // 2, act=act, shortcut=True) for i in range(n)]) #n Res Block
        if attn: #effective SE
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None
        self.conv3 = ConvBNLayer(ch_mid, c2, 1, act=act)#CBR

        if infor == "neck":
            _c2 = c2//2
            self.conv1 = ConvBNLayer(c1, _c2, 1, act=act)
            self.conv2 = ConvBNLayer(c1, _c2, 1, act=act)
            self.attn = None #neck中无effective SE
            self.conv3 = ConvBNLayer(c2, c2, 1, act=act)
            self.blocks = nn.Sequential(*[block_fn(_c2, _c2, act=act, shortcut=False) for i in range(n)])  # n Res Block,no shortcut in neck

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y

# 舍弃，用了新版的RepVggConv
# class RepVggBlock(nn.Module):
#     def __init__(self, ch_in, ch_out, act='relu', deploy=False):
#         super(RepVggBlock, self).__init__()
#         self.ch_in = ch_in
#         self.ch_out = ch_out
#         self.deploy = deploy
#         if self.deploy == False:
#             self.conv1 = ConvBNLayer(
#                 ch_in, ch_out, 3, stride=1, padding=1, act=None)
#             self.conv2 = ConvBNLayer(
#                 ch_in, ch_out, 1, stride=1, padding=0, act=None)
#         else:
#             self.conv = nn.Conv2d(
#                 in_channels=self.ch_in,
#                 out_channels=self.ch_out,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 groups=1
#             )
#         self.act = get_activation(act) if act is None or isinstance(act, (
#             str, dict)) else act
#
#     def forward(self, x):
#         if self.deploy:
#             y = self.conv(x)
#         else:
#             y = self.conv1(x) + self.conv2(x)
#         y = self.act(y)
#         return y
#
#     def switch_to_deploy(self):
#         if not hasattr(self, 'conv'):
#             self.conv = nn.Conv2d(
#                 in_channels=self.ch_in,
#                 out_channels=self.ch_out,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 groups=1
#             )
#         kernel, bias = self.get_equivalent_kernel_bias()
#         self.conv.weight.data = kernel
#         self.conv.bias.data = bias
#         for para in self.parameters():
#             para.detach_()
#         self.__delattr__(self.conv1)
#         self.__delattr__(self.conv2)
#         self.deploy = True
#
#     def get_equivalent_kernel_bias(self):
#         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
#         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
#         return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1
#
#     def _pad_1x1_to_3x3_tensor(self, kernel1x1):
#         if kernel1x1 is None:
#             return 0
#         else:
#             return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
#
#     def _fuse_bn_tensor(self, branch):
#         if branch is None:
#             return 0, 0
#         if isinstance(branch, nn.Sequential):
#             kernel = branch.conv.weight
#             running_mean = branch.bn.running_mean
#             running_var = branch.bn.running_var
#             gamma = branch.bn.weight
#             beta = branch.bn.bias
#             eps = branch.bn.eps
#         else:
#             assert isinstance(branch, nn.BatchNorm2d)
#             if not hasattr(self, 'id_tensor'):
#                 input_dim = self.in_channels // self.groups
#                 kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
#                 for i in range(self.in_channels):
#                     kernel_value[i, i % input_dim, 1, 1] = 1
#                 self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
#             kernel = self.id_tensor
#             running_mean = branch.running_mean
#             running_var = branch.running_var
#             gamma = branch.weight
#             beta = branch.bias
#             eps = branch.eps
#         std = (running_var + eps).sqrt()
#         t = (gamma / std).reshape(-1, 1, 1, 1)
#         return kernel * t, beta - running_mean * gamma / std

class RepVggBlock(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


#EffectiveSELayer
class EffectiveSELayer(nn.Module):
    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = get_activation(act) if act is None or isinstance(act, (str, dict)) else act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)

#Res Blocks = CBS + RepVGG Block  concat
class BasicBlock(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 act='relu',
                 shortcut=True,
                 use_alpha=False):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x+y
        else:
            return y

def identity(x):
    return x

__all__ = [

    'mish',
    'silu',
    'swish',
    'identity',
]
def mish(x):
    return nn.mish(x)

def relu(x):
    return nn.relu(x)
def silu(x):
    return nn.silu(x)


def swish(x):
    return x * nn.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name is None:
        return nn.Identity()

    if isinstance(name, str):
        if name == "silu":
            module = nn.SiLU(inplace=inplace)
        elif name == "relu":
            module = nn.ReLU(inplace=inplace)
        elif name == "lrelu":
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'swish':
            module = Swish(inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        else:
            raise AttributeError("Unsupported act type: {}".format(name))
        return module
    elif isinstance(name, nn.Module):
        return name
    else:
        raise AttributeError("Unsupported act type: {}".format(name))

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


#--------------------------------------PP-YOLO START------------------------------------------------------
def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding



def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_prob=0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]




def make_blocks(
        block_fn, idx,bool_DeformableConv2d,channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(idx,net_block_idx,bool_DeformableConv2d,
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info



def add_coord(x):
    ins_feat = x  # 当前实例特征tensor
    # 生成从-1到1的线性值
    x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
    y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
    y, x = torch.meshgrid(y_range, x_range)  # 生成二维坐标网格
    y = y.expand([ins_feat.shape[0], 1, -1, -1])  # 扩充到和ins_feat相同维度
    x = x.expand([ins_feat.shape[0], 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)  # 位置特征
    ins_feat = torch.cat([ins_feat, coord_feat], 1)  # concatnate一起作为下一个卷积的输入

    return ins_feat
class CoordConv(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 padding,
                 data_format='NCHW'):
        """
        CoordConv layer, see https://arxiv.org/abs/1807.03247

        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            filter_size (int): filter size, default 3
            padding (int): padding size, default 0
            norm_type (str): batch norm type, default bn
            name (str): layer name
            data_format (str): data format, NCHW or NHWC

        """
        super(CoordConv, self).__init__()
        self.conv = Conv(
            ch_in + 2,
            ch_out,
            k=filter_size,
            p=padding)
        self.data_format = data_format

    def forward(self, x):
        ins_feat = add_coord(x)

        y = self.conv(ins_feat)
        return y

class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=None,
        *,
        offset_groups=1,
        with_mask=True,
    ):
        super().__init__()
        assert in_dim % groups == 0
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.bias = None

        self.with_mask = with_mask
        if with_mask:
            # batch_size, (2+1) * offset_groups * kernel_height * kernel_width, out_height, out_width
            self.param_generator = nn.Conv2d(in_dim, 3 * offset_groups * kernel_size * kernel_size, 3, stride, 1)
        else:
            self.param_generator = nn.Conv2d(in_dim, 2 * offset_groups * kernel_size * kernel_size, 3, 1, 1)

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.param_generator(x)
            mask = None
        x = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return x

def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)

class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

    Parameters
    ----------
    block : Block, class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int, number of layers in each block
    num_classes : int, default 1000, number of classification classes.
    in_chans : int, default 3, number of input (color) channels.
    output_stride : int, default 32, output stride of the network, 32, 16, or 8.
    global_pool : str, Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    cardinality : int, default 1, number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64, factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64, number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first : int, default 1
        Reduction factor for first convolution output width of residual blocks, 1 for all archs except senets, where 2
    down_kernel_size : int, default 1, kernel size of residual block downsample path, 1x1 for most, 3x3 for senets
    avg_down : bool, default False, use average pooling for projection skip connection between stages/downsample.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0. Dropout probability before classifier, for training
    """

    def __init__(
            self, cout = 64,idx = 0,bool_DeformableConv2d = False,channels = [64, 128, 256, 512],block = Bottleneck, layers=[3, 4, 6, 3], num_classes=1000, in_chans=3, output_stride=32, global_pool='avg',
            cardinality=1, base_width=64, stem_width=64, stem_type='', replace_stem_pool=False, block_reduce_first=1,
            down_kernel_size=1, avg_down=True, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None,
            drop_rate=0.0, drop_path_rate=0., drop_block_rate=0., zero_init_last=True, block_args=None):
        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.idx = idx
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.bool_DeformableConv2d = bool_DeformableConv2d

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True)
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        self.channels = channels
        stage_modules, stage_feature_info = make_blocks(
            block,idx,bool_DeformableConv2d,channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion


        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    # def forward_features(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.act1(x)
    #     x = self.maxpool(x)
    #
    #
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        if self.idx == 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.maxpool(x)
        if self.idx == 1:
            x = self.layer1(x)
        if self.idx ==2:
            x = self.layer2(x)
        if self.idx == 3:
            x = self.layer3(x)
        if self.idx ==4:
            x = self.layer4(x)


        return x
layers=[3, 4, 6, 3]

class ResNet50vd_samll(nn.Module):
    def __init__(self,cout = 64,idx = 0):

        super(ResNet50vd , self).__init__()
        self.cout = cout
        self.idx = idx
        self.resnet50vd  = ResNet(channels = [32, 64, 128, 256],cout = cout,idx = idx,block=Bottleneck, layers=layers, stem_width=32, stem_type='deep', avg_down=True,bool_DeformableConv2d = False)
    def forward(self, x):
        x = self.resnet50vd(x)
        return x

class ResNet50vd(nn.Module):
    def __init__(self,cout = 64,idx = 0):

        super(ResNet50vd , self).__init__()
        self.cout = cout
        self.idx = idx
        self.resnet50vd  = ResNet(channels = [64, 128, 256, 512],cout = cout,idx = idx,block=Bottleneck, layers=layers, stem_width=32, stem_type='deep', avg_down=True,bool_DeformableConv2d = False)
    def forward(self, x):
        x = self.resnet50vd(x)
        return x

class ResNet50vd_dcn(nn.Module):

    def __init__(self,cout = 64,idx = 0):

        super(ResNet50vd_dcn , self).__init__()
        self.cout = cout
        self.idx = idx
        self.resnet50vd_dcn  = ResNet(channels = [64, 128, 256, 512],cout = cout,idx = idx,block=Bottleneck, layers=layers, stem_width=32, stem_type='deep', avg_down=True,bool_DeformableConv2d = True)
    def forward(self, x):
        x = self.resnet50vd_dcn(x)
        return x

class ResNet101vd(nn.Module):

    def __init__(self,cout = 64,idx = 0):

        super(ResNet101vd , self).__init__()
        self.cout = cout
        self.idx = idx
        self.resnet101vd  = ResNet(channels = [64, 128, 256, 512],cout = cout,idx = idx,block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True,bool_DeformableConv2d = False)
    def forward(self, x):
        x = self.resnet101vd(x)
        return x

class PPConvBlock(nn.Module):
    def __init__(self, channel,dropblock = False,coorConv = False):
        super(PPConvBlock, self).__init__()
        c_ = 2*channel
        self.channel = channel
        self.dropblock = dropblock
        self.coorConv = coorConv
        self.conv1 = Conv(channel,c_,3,1,1)
        self.conv2 = Conv(c_, channel, 1)
        if dropblock:
            self.drop = DropBlock2d(0.1, 3)
        if coorConv:
            self.conv2 = CoordConv(c_, channel, 1,0)
    def forward(self, x):

        x = self.conv1(x)
        if self.dropblock:
            x = self.drop(x)
        x = self.conv2(x)
        return x

class PPConvout(nn.Module):
    def __init__(self, channel):
        super(PPConvout, self).__init__()
        c_ = 2*channel
        self.channel = channel
        self.conv1 = Conv(channel,c_,3,1,1)

    def forward(self, x):
        x = self.conv1(x)
        return x




#--------------------------------------PP-YOLO END------------------------------------------------------