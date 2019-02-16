#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:48:13 2019

@author: ubuntu
"""



def gen_base_anchors(anchor_base, anchor_ratios, anchor_scales):
    """生成9个base_anchors: [xmin,ymin,xmax,ymax]
        xmin = x_center - 
        ymin = 
        xmax = 
        ymax = 
    """
    ratios = torch.tensor(anchor_ratios)
    scales = torch.tensor(anchor_scales)
    
    w = anchor_base
    h = anchor_base
    x_ctr = w*0.5
    y_ctr = h*0.5
    h_ratios = torch.sqrt(ratios)
    w_ratios = 1/h_ratios
    
    ws1 = w * w_ratios[:, None] * scales[None, :]
    ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)

    base_anchors = torch.stack(
    [
        x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
        x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
    ],
    dim=-1).round()
    
    return base_anchors


def gen_base_anchors_mine(anchor_base, ratios, scales):
    """生成9个base anchors, [xmin,ymin,xmax,ymax]
    Args:
        anchor_base(float): 表示anchor的基础尺寸
        ratios(list(float)): 表示h/w，由于r=h/w, 所以可令h'=sqrt(r), w'=1/sqrt(r), h/w就可以等于r了
        scales(list(float)): 表示整体缩放倍数
    1. 计算h, w
        h = base * scale * sqrt(ratio)
        w = base * scale * sqrt(1/ratio)
    2. 计算坐标
        xmin = x_center - w/2
        ymin = y_center - h/2
        xmax = x_center + w/2
        ymax = y_center + h/2
        
    """
    ratios = torch.tensor(ratios) 
    scales = torch.tensor(scales)
    w = anchor_base
    h = anchor_base
    x_ctr = 0.5 * w
    y_ctr = 0.5 * h
    
    base_anchors = torch.zeros(len(ratios)*len(scales),4)   # (n, 4)
    for i in range(len(ratios)):
        for j in range(len(scales)):
            h = anchor_base * scales[j] * torch.sqrt(ratios[i])
            w = anchor_base * scales[j] * torch.sqrt(1. / ratios[i])
            index = i*len(scales) + j
            base_anchors[index, 0] = x_ctr - 0.5 * w  # 
            base_anchors[index, 1] = y_ctr - 0.5 * h
            base_anchors[index, 2] = x_ctr + 0.5 * w
            base_anchors[index, 3] = y_ctr + 0.5 * h
    
    return base_anchors.round()

def _meshgrid(x, y, row_major=True):
    xx = x.repeat(len(y))
    yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx
    
def grid_anchors(featmap_size, stride, base_anchors, device='cpu'):
    base_anchors = base_anchors.to(device)
#
    feat_h, feat_w = featmap_size
    shift_x = torch.arange(0, feat_w, device=device) * stride  # 256
    shift_y = torch.arange(0, feat_h, device=device) * stride  # 152
    shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
    shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
    shifts = shifts.type_as(base_anchors)
    # first feat_w elements correspond to the first row of shifts
    # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
    # shifted anchors (K, A, 4), reshape to (K*A, 4)

    all_anchors = base_anchors[None, :, :] + shifts[:, None, :]  # (1,9,4) + (38912,1,4) = (38912, 9, 4)
    all_anchors = all_anchors.view(-1, 4)
    # first A rows correspond to A anchors of (0, 0) in feature map,
    # then (0, 1), (0, 2), ...
    return all_anchors


def grid_anchors_mine(featmap_size, stride, base_anchors):
    """基于base anchors把特征图的每个网格都放置anchors
    Args:
        featmap_size(list(float))
        stride(float): 代表该特征图相对于原图的下采样比例，也就代表每个网格的感受野是多少尺寸的原图网格，比如1个就相当与stride x stride大小的一片原图
        device
    1. 先计算该特征图对应原图像素大小 = 特征图大小 x 下采样比例
    2. 然后生成网格坐标xx, yy并展平：先得到x坐标，再meshgrid思想得到网格xx坐标，再展平
       其中x坐标就是按照采样比例，每隔1个stride取一个坐标
    3. 然后堆叠出[xx,yy,xx,yy]分别叠加到anchor原始坐标[xmin,ymin,xmax,ymax]上去(最难理解，广播原则)
    4. 最终得到特征图上每个网格点上都安放的n个base_anchors
    """
    feat_h, feat_w = featmap_size
    shift_x = torch.arange(0, feat_w) * stride  # 先放大到原图大小 (256)
    shift_y = torch.arange(0, feat_h) * stride  #                 (152)
    shift_xx = shift_x[None,:].repeat((len(shift_y), 1))   # (152,256)
    shift_yy = shift_y[:, None].repeat((1, len(shift_x)))  # (152,256)
    
    shift_xx = shift_xx.flatten()   # (38912,) 代表了原始图的每个网格点x坐标，用于给x坐标平移
    shift_yy = shift_yy.flatten()   # (38912,) 代表了原始图的每个网格点y坐标，用于给y坐标平移
    
    shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1) # 堆叠成4行给4个坐标[xmin,ymin,xmax,ymax], (38912,4)
    shifts = shifts.type_as(base_anchors)   # 从int64转换成torch认可的float32
    
    # anchor坐标(9,4)需要基于网格坐标(38912,4)进行平移：平移后应该是每个网格点有9个anchor
    # 也就是38912个(9,4)，也就等效于anchor(9,4)与每一个网格坐标(1,4)进行相加
    # 需要想到把(38912,4)提取出(1,4)的方式是升维到(38912,1,4)与(9,4)相加
    all_anchors = base_anchors + shifts[:,None,:]   # 利用广播法则(9,4)+(38912,1,4)->(39812,9,4)
    all_anchors = all_anchors.view(-1,4)            # 部分展平到(n,4)得到每个anchors的实际坐标(图像左上角为(0,0)原点)                      

    return all_anchors
    
    
import torch    
anchor_strides = [4., 8., 16., 32., 64.]
anchor_base_sizes = anchor_strides      # 基础尺寸
anchor_scales = [8., 16., 32.]          # 缩放比例
anchor_ratios = [0.5, 1.0, 2.0]         # w/h比例

num_anchors = len(anchor_scales) * len(anchor_ratios)
base_anchors = []
for anchor_base in anchor_base_sizes:
    base_anchors.append(gen_base_anchors_mine(anchor_base, anchor_ratios, anchor_scales))


featmap_sizes = [(152,256), (76,128), (38,64), (19,32), (10,16)]
strides = [4,8,16,32,64]    # 针对resnet的下采样比例，5路分别缩减尺寸 

i=0
featmap_size = featmap_sizes[i]
stride = strides[i]
base_anchor = base_anchors[i]
all_anchors1 = grid_anchors(featmap_size, stride, base_anchor)
all_anchors2 = grid_anchors_mine(featmap_size, stride, base_anchor)





