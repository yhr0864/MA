# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math
from collections import OrderedDict

import numpy as np
import torch.nn.functional as F

from general_functions.loss import compute_loss
from general_functions.quan import QuanConv2d, QuanAct, quantizer
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

import torch
import torch.nn as nn
from .layers import (BatchNorm2d, Conv2d, FrozenBatchNorm2d, interpolate)
from .modeldef import MODEL_ARCH, Test_model_arch

logger = logging.getLogger(__name__)


def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret


# include all the operations
PRIMITIVES = {
   "none": lambda C_in, C_out, stride, prune, **kwargs: Zero(
        stride
    ),
    "skip": lambda C_in, C_out, stride, prune, **kwargs: Identity(
        C_in[0], C_out, stride
    ),
    "CBL_k3": lambda C_in, C_out, stride, prune, **kwargs: ConvBNRelu(
        C_in[0], C_out, kernel=3, stride=1, pad=1, use_relu="leaky", bn_type="bn", **kwargs
    ),
    "CBL_k5": lambda C_in, C_out, stride, prune, **kwargs: ConvBNRelu(
        C_in[0], C_out, kernel=5, stride=1, pad=2, use_relu="leaky", bn_type="bn", **kwargs
    ),
    "CBL_k7": lambda C_in, C_out, stride, prune, **kwargs: ConvBNRelu(
        C_in[0], C_out, kernel=7, stride=1, pad=3, use_relu="leaky", bn_type="bn", **kwargs
    ),
    "ir_k3_re": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, expansion, prune, kernel=3, nl="relu", **kwargs
    ),
    "ir_k3_re_e1": lambda C_in, C_out, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, prune, kernel=3, nl="relu", expansion=1, **kwargs
    ),
    "ir_k3_re_e3": lambda C_in, C_out, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, prune, kernel=3, nl="relu", expansion=3, **kwargs
    ),
    "ir_k3_re_e6": lambda C_in, C_out, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, prune, kernel=3, nl="relu", expansion=6, **kwargs
    ),
    "ir_k3_hs": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, expansion, prune, kernel=3, nl="hswish", **kwargs
    ),
    "ir_k3_r2_re": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, expansion, prune, kernel=3, nl="relu", dil=2, **kwargs
    ),
    "ir_k3_r2_hs": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, expansion, prune, kernel=3, nl="hswish", dil=2, **kwargs
    ),
    "ir_k3_r3_re": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, expansion, prune, kernel=3, nl="relu", dil=3, **kwargs
    ),
    "ir_k5_re": lambda C_in, C_out, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, prune, kernel=5, nl="relu", **kwargs
    ),
    "ir_k5_re_e1": lambda C_in, C_out, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, prune, kernel=5, nl="relu", expansion=1, **kwargs
    ),
    "ir_k5_re_e3": lambda C_in, C_out, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, prune, kernel=5, nl="relu", expansion=3, **kwargs
    ),
    "ir_k5_re_e6": lambda C_in, C_out, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, prune, kernel=5, nl="relu", expansion=6, **kwargs
    ),
    "ir_k5_hs": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, expansion, prune, kernel=5, nl="hswish", **kwargs
    ),
    "ir_k5_r2_re": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, expansion, prune, kernel=5, nl="relu", dil=2, **kwargs
    ),
    "ir_k5_r2_hs": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, expansion, prune, kernel=5, nl="hswish", dil=2, **kwargs
    ),
    "ir_k5_r3_re": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, expansion, prune, kernel=5, nl="relu", dil=3, **kwargs
    ),
    "ir_k7_re": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, expansion, prune, kernel=7, nl="relu", **kwargs
    ),
    "ir_k7_re_e1": lambda C_in, C_out, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, prune, kernel=7, nl="relu", expansion=1, **kwargs
    ),
    "ir_k7_re_e3": lambda C_in, C_out, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, prune, kernel=7, nl="relu", expansion=3, **kwargs
    ),
    "ir_k7_re_e6": lambda C_in, C_out, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, prune, kernel=7, nl="relu", expansion=6, **kwargs
    ),
    "ir_k7_hs": lambda C_in, C_out, expansion, stride, prune, **kwargs: IRFBlock(
        C_in[0], C_out, stride, expansion, prune, kernel=7, nl="hswish", **kwargs
    ),
}


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

    @property
    def module_list(self):
        return False


class Identity(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Identity, self).__init__()
        self.output_depth = C_out
        self.moduleList = nn.ModuleList([
            ConvBNRelu(
                C_in,
                C_out,
                kernel=1,
                stride=stride,
                pad=0,
                
                use_relu="relu",
                bn_type="bn",
            )]) if C_in != C_out or stride != 1 else None

    def forward(self, x):
        if self.moduleList:
            out = self.moduleList[0](x)
        else:
            out = x
        return out


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
                .view(N, C, H, W)
        )


class ConvBNRelu(nn.Sequential):
    def __init__(
            self,
            input_depth,
            output_depth,
            kernel,
            stride,
            pad,
            use_relu,
            bn_type,
            group=1,
            dil=1,
            bias=False,
            quant=False,
            *args,
            **kwargs
    ):
        super(ConvBNRelu, self).__init__()

        assert use_relu in ["relu", "hswish", "leaky", None, False]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ["bn", "af", "gn", None]
        assert stride in [1, 2, 4]
        assert dil in [1, 2, 3, None]

        op = nn.Conv2d(
            input_depth,
            output_depth,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            dilation=dil,
            bias=bias,
            groups=group,
            *args,
            **kwargs
        )
        
        if quant:
            op = QuanConv2d(op,
                            quan_w_fn=quantizer(CONFIG_SUPERNET['quan']['weight']),
                            quan_a_fn=quantizer(CONFIG_SUPERNET['quan']['act'])) 
        
        self.add_module("conv", op)

        if bn_type == "bn":
            bn_op = nn.BatchNorm2d(output_depth)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth)
        elif bn_type == "af":
            bn_op = FrozenBatchNorm2d(output_depth)
        if bn_type is not None:
            self.add_module("bn", bn_op)

        if use_relu == "relu":
            act = nn.ReLU(inplace=True)
            self.add_module("act", act)
        elif use_relu == "hswish":
            act = nn.Hardswish(inplace=True)
            self.add_module("act", act)
        elif use_relu == "leaky":
            act = nn.LeakyReLU(0.1, inplace=True)
            self.add_module("act", act)


class SEModule(nn.Module):
    reduction = 4

    def __init__(self, C):
        super(SEModule, self).__init__()
        mid = max(C // self.reduction, 8)
        conv1 = Conv2d(C, mid, 1, 1, 0)
        conv2 = Conv2d(mid, C, 1, 1, 0)

        self.op = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), conv1, nn.ReLU(inplace=True), conv2, nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.op(x)


# class Upsample(nn.Module):
#     def __init__(self, scale_factor, mode, align_corners=None):
#         super(Upsample, self).__init__()
#         self.scale = scale_factor
#         self.mode = mode
#         self.align_corners = align_corners

#     def forward(self, x):
#         return interpolate(
#             x, scale_factor=self.scale, mode=self.mode,
#             align_corners=self.align_corners
#         )

#     @property
#     def module_list(self):
#         return False

    
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
    
    @property
    def module_list(self):
        return False
    

class IRFBlock(nn.Module):
    def __init__(
            self,
            input_depth,
            output_depth,
            stride,
            prune=None,
            bn_type="bn",
            kernel=3,
            nl="relu",
            expansion=None,
            dil=1,
            width_divisor=1,
            shuffle_type=None,
            pw_group=1,
            se=False,
            dw_skip_bn=False,
            dw_skip_relu=False
    ):
        super(IRFBlock, self).__init__()

        assert kernel in [1, 3, 5, 7], kernel

        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth

        self.module_list = nn.ModuleList()

        if expansion:
            mid_depth = int(input_depth * expansion)
            mid = mid_depth
        elif prune:
            mid_depth = prune[0]
            mid = prune[1]
        else:
            raise ValueError("neither given expansion nor mid")

        mid_depth = _get_divisible_by(mid_depth, width_divisor, width_divisor)

        # pw
        self.pw = ConvBNRelu(
            input_depth,
            mid_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=nl,
            bn_type=bn_type,
            group=pw_group,
        )
        self.module_list.append(self.pw)
        # dw
        self.dw = ConvBNRelu(
            mid_depth,
            mid,
            kernel=kernel,
            stride=stride,
            pad=(kernel // 2) * dil,
            dil=dil,
            group=mid_depth,
            no_bias=1,
            use_relu=nl if not dw_skip_relu else None,
            bn_type=bn_type if not dw_skip_bn else None,
        )
        self.module_list.append(self.dw)
        # pw-linear
        self.pwl = ConvBNRelu(
            mid,
            output_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=False,
            bn_type=bn_type,
            group=pw_group,
        )
        self.module_list.append(self.pwl)

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        self.se4 = SEModule(output_depth) if se else nn.Sequential()

        self.output_depth = output_depth

    def forward(self, x):
        y = self.module_list[0](x) # pw
        if self.shuffle_type == "mid":
            y = self.shuffle(y)

        y = self.module_list[1](y) # dw
        y = self.module_list[2](y) # pwl
        if self.use_res_connect:
            y += x
        #y = self.se4(y)
        return y


class SampledNet(nn.Module):
    def __init__(self, arch_def, num_anchors, num_cls,
                 layer_parameters,
                 layer_parameters_head26,
                 layer_parameters_head13,
                 
                 yolo_layer26,
                 yolo_layer13,
                 prune_para=None):
        super(SampledNet, self).__init__()
        self.module_list = nn.ModuleList()
        self.first = ConvBNRelu(input_depth=3, output_depth=16, kernel=3, stride=1,
                                pad=3 // 2, use_relu="leaky", bn_type="bn")

        self.module_list.append(self.first) # i=0
        self.module_list.append(nn.MaxPool2d(2, stride=2))

        operations = lambda x: {op_name[0]: PRIMITIVES[op_name[0]] for op_name in x}

        self.backbones = arch_def['block_op_type_backbone']
        self.head26 = arch_def['block_op_type_head26']
        self.head13 = arch_def['block_op_type_head13']

        for i, op_name in enumerate(self.backbones):
            self.module_list.append(operations(self.backbones)[op_name[0]](*layer_parameters[i]))  # i=1~11
            
            if i < 4:
                self.module_list.append(nn.MaxPool2d(2, stride=2))

            elif i == 4:
                self.module_list.append(nn.MaxPool2d(2, stride=1))
            
        # preprocess: input_depth = layer_parameters[-1][1]
        #             output_depth = layer_parameters_fpn[0][1] 与fpn所有同步
        
        if prune_para is None:
            self.conv1x1_1 = ConvBNRelu(input_depth=1024,
                                    output_depth=256,
                                    kernel=1, stride=1,
                                    pad=0, use_relu="leaky", bn_type="bn")
            
            self.conv1x1_2 = ConvBNRelu(input_depth=256,
                                    output_depth=128,
                                    kernel=1, stride=1,
                                    pad=0, use_relu="leaky", bn_type="bn")
            
        else:           
            self.conv1x1_1 = ConvBNRelu(input_depth=layer_parameters[-1][1],
                                    output_depth=prune_para[0][0],
                                    kernel=1, stride=1,
                                    pad=0, use_relu="leaky", bn_type="bn")

            self.conv1x1_2 = ConvBNRelu(input_depth=prune_para[0][0],
                                    output_depth=prune_para[1][0],
                                    kernel=1, stride=1,
                                    pad=0, use_relu="leaky", bn_type="bn")
            
        self.module_list.append(self.conv1x1_1) # i=13
        self.module_list.append(self.conv1x1_2)

        self.upsample = Upsample(scale_factor=2, mode="nearest")
        self.module_list.append(self.upsample) # i=15

        for i, op_name in enumerate(self.head26):
            self.module_list.append(operations(self.head26)[op_name[0]](*layer_parameters_head26[i])) 

        for i, op_name in enumerate(self.head13):
            self.module_list.append(operations(self.head13)[op_name[0]](*layer_parameters_head13[i]))  
        
        self.head_converter26 = ConvBNRelu(input_depth=layer_parameters_head26[-1][1],
                                           output_depth=num_anchors * (num_cls + 5),
                                           kernel=1, stride=1, pad=0, 
                                           use_relu=None, bn_type=None, bias=False)

        self.head_converter13 = ConvBNRelu(input_depth=layer_parameters_head13[-1][1],
                                           output_depth=num_anchors * (num_cls + 5),
                                           kernel=1, stride=1, pad=0, 
                                           use_relu=None, bn_type=None, bias=False)

        self.module_list.append(self.head_converter26)  # i=35
        self.module_list.append(self.head_converter13)  # i=36

        self.yololayer_26 = yolo_layer26
        self.yololayer_13 = yolo_layer13

        self.module_list.append(self.yololayer_26) # i=37
        self.module_list.append(self.yololayer_13) # i=38

        self.yolo_layers = [self.yololayer_26, self.yololayer_13]

    def forward(self, x):
        img_size = x.size(2)

        # first layer
        x = self.module_list[0](x)
        y = self.module_list[1](x) # dowmsample
        
        # backbones
        for i in range(len(self.backbones)*2-1): # i=1~11
            if i % 2 == 0:
                y = self.module_list[i+2](y)
                if i == 6: # FPN26
                    fpn26 = y
            else:
                if i == 9:
                    y = nn.ZeroPad2d((0, 1, 0, 1))(y)
                y = self.module_list[i+2](y) # dowm sample
                
        fpn13 = y
        
        # fpn13 ch: 1024 -> 256
        id_conv1x1_1 = 2 + len(self.backbones)*2-1

        fpn13 = self.module_list[id_conv1x1_1](fpn13) # i=13

        # FPN
        id_conv1x1_2 = id_conv1x1_1 + 1 # i=14
        id_upsample = 1 + id_conv1x1_2 # i=15

        hid_layer13 = self.module_list[id_conv1x1_2](fpn13)
        hid_layer13 = self.module_list[id_upsample](hid_layer13)

        fpn26 = torch.cat((hid_layer13, fpn26), 1)

        # head
        id_head26 = id_upsample + 1 # i=16
        id_head13 = id_head26 + len(self.head26) # i=17
        
        for i in range(len(self.head26)):
            fpn26 = self.module_list[id_head26+i](fpn26) # i=25~29
            fpn13 = self.module_list[id_head13+i](fpn13) # i=30~34

        # yolo_layer
        id_head_converter26 = id_head13 + len(self.head13) # i=35
        id_head_converter13 = id_head_converter26 + 1  # i=36

        yololayer_26 = self.module_list[id_head_converter26](fpn26)
        yololayer_13 = self.module_list[id_head_converter13](fpn13)

        id_yololayer26 = id_head_converter13 + 1 # i=37
        id_yololayer13 = id_yololayer26 + 1 # i=38

        output26 = self.module_list[id_yololayer26](yololayer_26, img_size)
        output13 = self.module_list[id_yololayer13](yololayer_13, img_size)
        yolo_output = [output26, output13]
        return yolo_output if self.training else torch.cat(yolo_output, 1)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outs, targets, model):
        ce, loss_components = compute_loss(outs, targets, model)
        return ce, loss_components


if __name__ == '__main__':
    #from supernet_functions.lookup_table_builder import (LookUpTable, SEARCH_SPACE_BACKBONE, SEARCH_SPACE_HEAD, SEARCH_SPACE_FPN,
    #                                                     YOLO_LAYER_26, YOLO_LAYER_13)
    #arch = 'test_net'
    #arch_def = Test_model_arch[arch]
    #print(arch_def['block_op_type_backbone'])
    #backbones = arch_def['block_op_type_backbone']

    #operations = {op_name[0]: PRIMITIVES[op_name[0]] for op_name in backbones}
   #print(operations)

    #layer_parameters, _ = LookUpTable._generate_layers_parameters(search_space_backbone=SEARCH_SPACE_BACKBONE)
    #print(*layer_parameters)
    #operations = lambda part: {op_name[0]: PRIMITIVES[op_name[0]] for op_name in part}

    #operations = {op_name[0]: PRIMITIVES[op_name[0]] for op_name in backbones}
    #print([operations(backbones)[op_name[0]](*layer_parameters[i]) for i, op_name in enumerate(backbones)])  # Modulelist for backbone
    # op = ConvBNRelu(
    #     3,
    #     10,
    #     kernel=1,
    #     stride=1,
    #     pad=1,
    #     no_bias=1,
    #     use_relu='relu',
    #     bn_type='bn'
    # )
    # for i, m in enumerate(op.modules()):
    #     print(i)
    #     print(m)
    # input = torch.randn(1,3,10,10)
    # out = op(input)
    # print(out)
    # convqt = QuanConv2d(Conv2d(3,10,1,1), quan_w_fn=quantizer(CONFIG_SUPERNET['quan']['weight']))
    # print("normal conv: ", Conv2d(3,10,1,1))
    # print("QT-conv: ", convqt)
    # #print(convqt(input))
    # act = nn.Hardswish(inplace=True)
    # actqt = QuanAct(act, quan_a_fn=quantizer(CONFIG_SUPERNET['quan']['act']))
    # input2 = torch.randn(5,5)
    # print(input2)
    # print(actqt(input2))
    input = torch.randn(1, 3, 320, 320)
    for k in PRIMITIVES:
        op = PRIMITIVES[k](3, 16, 6, 1)
        #print(op)
        torch.onnx.export(op,  # model being run
                          input,  # model input (or a tuple for multiple inputs)
                          "./onnx/{}.onnx".format(k),  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          # the ONNX version to export the model to
                          opset_version=7,
                          verbose=True,
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes=None,
                         )
