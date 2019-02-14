#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:48:13 2019

@author: ubuntu
"""
import torch
import torch.nn as nn
import torch.functional as F
# FPN的实现方式: 实现一个简单的FPN结构
class FPN_neck():
    def __init__(self):
        self.outs_layers = 5                  # 定义需要FPN输出的特征层数
        self.lateral_conv = nn.ModuleList()   # 横向卷积1x1，用于调整层数为统一的256
        self.fpn_conv = nn.ModuleList()       # FPN卷积3x3，用于抑制中间上采样之后的两层混叠产生的混叠效应
        in_channels = [256,512,1024,2048]
        out_channels = 256
        for i in range(5):
            lc = nn.Sequential(nn.Conv2d(in_channels[i],out_channels, 1, 1, 0),  # 1x1保证尺寸不变，层数统一到256
                               nn.BatchNorm2d(),
                               nn.ReLU())
            fc = nn.Sequential(nn.Conv2d(in_channels[i],out_channels,1, 1, 1),   # 3x3保证尺寸不变。注意s和p的参数修改来保证尺寸不变
                               nn.BatchNorm2d(),
                               nn.ReLU())
            self.lateral_conv.append(lc)
            self.fpn_conv.append(fc)
    def forward(self, feats):
        lateral_outs = []
        for i, lc in enumerate(self.lateral_conv):
            lateral_outs.append(lc(feats[i]))       # 获得横向输出
        for i in range(2,0,-1):                     # 进行上采样和混叠
            lateral_outs[i] += F.interpolate(lateral_outs[i+1], scale_factor=2, mode='nearest')
        lateral_outs.append(nn.MaxPool2d())         # 增加一路maxpool输出
        outs = []
        for i, fc in enumerate(self.fpn_conv):      # 获得fpn卷积层的输出
            outs.append(fc(lateral_outs[i]))
        if len(outs) < self.outs_layers:                 # 如果需要的输出特征图层超过当前backbone的输出，则用maxpool替代
            for i in len(self.outs_layers - len(outs)):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return outs
    
feats_channels = [256,512,1024,2048]
feats_sizes = [(152,256),(76,128),(38,64),(19,32)]
feats = []
for i in range(4):  # 构造假数据
    feats.append(torch.randn(2,feats_channels[i],feats_sizes[i][0],feats_sizes[i][1]))
fpn = FPN_neck()
outs = fpn(feats)