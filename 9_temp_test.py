#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:48:13 2019

@author: ubuntu
"""

import torch
import numpy as np

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

def bbox_overlap(bboxes1,bboxes2,mode='iou'):

    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # (m, 1, 2) vs (n, 2) -> (m,n,2) 代表xmin,ymin
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # (m, 1, 2) vs (n, 2) -> (m,n,2) 代表xmax,ymax

    wh = (rb - lt + 1).clamp(min=0)                       # (m,n,2) - (m,n,2) = (m,n,2) 代表w,h
    overlap = wh[:, :, 0] * wh[:, :, 1]                   # (m,n) * (m,n)
    if mode == 'iou':
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
        ious = overlap / (area1[:, None] + area2 - overlap)
    return ious

def bbox_overlap_mine(bb1, bb2, mode='iou'):
    """bbox的重叠iou计算：iou = Intersection-over-Union交并比(假定bb1为gt_bboxes)
       还有一个iof = intersection over foreground就是交集跟gt_bbox的比
    Args:
        bb1(tensor): (m, 4) [xmin,ymin,xmax,ymax]
        bb2(tensor): (n, 4) [xmin,ymin,xmax,ymax]
    1. 计算两个bbox面积：area = (xmax - xmin)*(ymax - ymin)
    2. 计算两个bbox交集：
        >关键是找到交集方框的xmin,ymin,xmax,ymax
        >交集方框的xmin,ymin就等于两个bbox的xmin,ymin取最大值，
         因为这个方框在两个bbox的中间，其取值也就是取大值(画一下不同类型的bbox就清楚了)
         交集方框的xmax,ymax就等于两个bbox的xmax,ymax取最小值，理由同上
        >需要注意要求每个gt_bbox(bb1)跟每个anchor(bb2)的ious, 从数据结构设计上是
         gt(m,4), anchor(n,4)去找max,min, 先取m的一行(m,1,4)与(n,4)比较，
         最后得到(m,n,4)就代表了m层gt,每层都是(n,4),为一个gt跟n个anchor的比较结果。
        >计算交集方框的面积，就是overlap
    3. 计算ious: ious = overlap/(area1+area2-overlap)
    4. 为了搞清楚整个高维矩阵的运算过程中升维/降维的过程，关键在于：
        >抓住[变量组数]m,n的含义，这是不会变的，m和n有时候是层/有时候是行/有时候是列，
         但不会变的是m肯定是gt组数和n肯定是anchor组数
        >抓住运算的目的：如果是按位计算，简单的算就可以了；
         但如果是m组数要轮流跟n组数做运算，那就肯定先升维+广播做一轮运算
        >抓住每一轮输出变量的维度，确保了解这个维度的含义(基于变量组数不变来理解)
    """
    area1 = (bb1[:,2] - bb1[:,0]) * (bb1[:,3] - bb1[:,1]) # (m,)
    area2 = (bb2[:,2] - bb2[:,0]) * (bb2[:,3] - bb2[:,1]) # (n,)
    
    xymin = torch.max(bb1[:, None, :2], bb2[:,:2])  # 由于m个gt要跟n个anchor分别比较，所以需要升维度
    xymax = torch.min(bb1[:, None, 2:], bb2[:,2:])  # 所以(m,1,2) vs (n,2) -> (m,n,2)
    wh = (xymax -xymin).clamp(0)   # 得到宽高w, h (m,n,2)
    
    overlap = wh[:,:,0] * wh[:,:,1]   # (m,n)*(m,n) -> (m,n),其中m个gt的n列w, 乘以m个gt的n列h
    
    ious = overlap / (area1[:, None] + area2 -overlap) # 由于m个gt的每一个面积都要跟n的每一个面积相加，要得到(m,n)的面积之和
                                                       # 所以需要升维(m,1)+(n)->(m,n), 然后(m,n)-(m,n)，以及(m,n)/(m,n)都可以操作
    return ious

def assigner(bboxes, gt_bboxes):
    """anchor指定器：用于区分anchor的身份是正样本还是负样本还是无关样本
    正样本标记为1, 负样本标记为0, 无关样本标记为-1
    Args:
        bboxes(tensor): (m,4)
        gt_bboxes(tensor): (n,4)
    1. 先创建空矩阵，值设为-1
    2. 
    
    """
    pos_iou_thr = 0.7  # 正样本阀值：iou > 0.7 就为正样本
    neg_iou_thr = 0.3  # 负样本阀值：iou < 0.3 就为负样本
    min_pos_iou = 0.3  # 
    overlaps = bbox_overlap_mine(gt_bboxes, bboxes) # (m,n)代表m个gt, n个anchors
    # 第一步：先创建一个与所有anchor对应的矩阵，取值-1(代表没有用的anchor)
    
    max_overlap, argmax_overlap = overlaps.max(dim=0)
    pass
    # 第二步：



def sampler():
    """anchor抽样器"""
    pass
    
def anchor_target_mine():
    pass


#-------------base anchors---------------------   
import torch    
anchor_strides = [4., 8., 16., 32., 64.]
anchor_base_sizes = anchor_strides      # 基础尺寸
anchor_scales = [8., 16., 32.]          # 缩放比例
anchor_ratios = [0.5, 1.0, 2.0]         # w/h比例

num_anchors = len(anchor_scales) * len(anchor_ratios)
base_anchors = []
for anchor_base in anchor_base_sizes:
    base_anchors.append(gen_base_anchors_mine(anchor_base, anchor_ratios, anchor_scales))

#-------------all anchors---------------------
featmap_sizes = [(152,256), (76,128), (38,64), (19,32), (10,16)]
strides = [4,8,16,32,64]    # 针对resnet的下采样比例，5路分别缩减尺寸 

i=0
featmap_size = featmap_sizes[i]
stride = strides[i]
base_anchor = base_anchors[i]
all_anchors1 = grid_anchors(featmap_size, stride, base_anchor)
all_anchors2 = grid_anchors_mine(featmap_size, stride, base_anchor)


"""
#-------------valid flag---------------------
feat_h, feat_w = (152,256)
valid_h, valid_w = (148,240)
num_base_anchors = 9
assert valid_h <= feat_h and valid_w <= feat_w

valid_x = torch.zeros(feat_w, dtype=torch.uint8)
valid_y = torch.zeros(feat_h, dtype=torch.uint8)
valid_x[:valid_w] = 1
valid_y[:valid_h] = 1
#valid_xx, valid_yy = _meshgrid(valid_x, valid_y)
valid_xx = valid_x.repeat((len(valid_y),1))
valid_yy = valid_y[:,None].repeat(1,len(valid_x))

valid = valid_xx & valid_yy   # (152, 256)
valid1 = valid[:,None].expand(152,9)
valid = valid[:, None].expand(
    valid.size(0), num_base_anchors).contiguous().view(-1)
"""
#-------------bbox ious---------------------
bb1 = torch.tensor([[-20.,-20.,20.,20.],[-30.,-30.,30.,30.]])
bb2 = torch.tensor([[-25.,-25.,25.,25.],[-15.,-15.,15.,15.],[-25,-25,50,50]])
ious1 = bbox_overlap(bb1,bb2)
ious2 = bbox_overlap_mine(bb1, bb2)

#-------------anchor target---------------------
# 开始anchor target: rpn head中使用的anchor target()基本沿用默认设置，sampling=True
sampling =True  # 调用anchor target时没指定就沿用默认设置
# 在anchor target single()中，先调用assign_and_sample()
# 获得assign result, sample result
import pickle
gt_bboxes = pickle.load(open('test/test_data/test9_bboxes.txt','rb'))  # gt bbox (m,4)
gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)
bboxes = all_anchors2  # (n,4)
ious = bbox_overlap_mine(gt_bboxes, bboxes)

assign_result = assigner(bboxes, gt_bboxes)

assign_result, sampling_result = assign_and_sample(anchors, gt_bboxes, None, None, cfg)
gt_bboxes_ignore = None
gt_labels = None
bbox_assigner.assign(bboxes, gt_bboxes, None, None)
bbox_sampler.sample(assign_result, bboxes, gt_bboxes, None)







