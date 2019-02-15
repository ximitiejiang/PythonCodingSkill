#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:48:13 2019

@author: ubuntu
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
# 创建一个RPN head: 用于rpn/faster rcnn，输入的特征来自FPN，通道数相同，所以分类回归只需要3个卷积层
class RPN_head(nn.Module):
    def __init__(self, num_anchors):
        super().__init__()
        self.num_anchors = num_anchors
        self.rpn_conv = nn.Conv2d(256, 256, 3, 1, 1)  #
        self.rpn_cls = nn.Conv2d(256, self.num_anchors*2, 1, 1, 0)     # 多分类：输出通道数就是2*anchors，因为每个像素位置会有2个anchors (如果是2分类，也就是采用sigmoid而不是softmax，则只需要anchors个通道)
        self.rpn_reg = nn.Conv2d(256, self.num_anchors*4, 1, 1, 0)     # bbox回归：需要回归x,y,w,h四个参数
    def forward_single(self,feat):  # 对单层特征图的转换
        ft_cls = []
        ft_reg = []
        rpn_ft = self.rpn_conv(feat)
        ft_cls = self.rpn_cls(F.relu(rpn_ft))  #
        ft_reg = self.rpn_reg(F.relu(rpn_ft))
        return ft_cls, ft_reg
    def forward(self,feats):       # 对多层特征图的统一转换
        map_results = map(self.forward_single, feats)
        return tuple(map(list, zip(*map_results)))  # map解包，然后zip组合成元组，然后转成list，然后放入tuple
                                                    # ([(2,18,h,w),(..),(..),(..),(..)], 
                                                    #  [(2,36,h,w),(..),(..),(..),(..)])
channels = 256
sizes = [(152,256), (76,128), (38,64), (19,32), (10,16)]
fpn_outs = []
for i in range(5):
    fpn_outs.append(torch.randn(2,channels,sizes[i][0],sizes[i][1]))
rpn = RPN_head(9)  # 每个数据网格点有9个anchors
results = rpn(fpn_outs)  # ([ft_cls0, ft_cls1, ft_cls2, ft_cls3, ft_cls4],
                         #  [ft_reg0, ft_reg1, ft_reg2, ft_reg3, ft_reg4])
