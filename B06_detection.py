#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:24:32 2019

@author: ubuntu

本部分所有模块统一在最下面进行验证

"""

# %%
"""视觉检测这块如何分方向：
1. 分类/classification: 只需要对整个图形类别分类，比如：这张图像有气球
2. 对象检测/object detectin：需要识别具体物体的类型和位置，比如：这张图形有7个气球，每个气球的位置框
3. 语义分割/Semantic segmentation：需要识别气球，还要把气球从背景中整体分离出来，比如：这张图片有7个气球，分成气球和背景两部分
4. 实例分割/Instance Segmentation：需要识别气球，还要吧每个气球都单独分离并给出每个气球像素，比如：这张图片有7个气球，分成7组独立气球和1组背景
可以认为从1到4难度逐渐加大。
参考：https://blog.csdn.net/qq_15969343/article/details/80167215
"""
            
# %%
'''Q. backbones有哪些，跟常规network有什么区别？
'''    
# resnet作为backbone的修改


# 普通VGG16： blocks = [2,2,3,3,3]每个block包含conv3x3+bn+relu，算上最后3层linear总共带参层就是16层，也就是vgg16名字由来
import torch.nn as nn
class VGG16(nn.Module):
    def __init__():
        super().__init__()
        self.blocks = [2,2,3,3,3]
        for i in self.blocks:
            layers.append(make_vgg_layer())

# SSD修改的VGG: 需要提取出6路多尺度的特征图
class SSD_VGG16(VGG16):
    def __init__():
        super().__init__()
        
    def forward(self, feats):
        pass

# %%
"""如何从backbone获得multi scale多尺度不同分辨率的输出？
1. 方式1：通过FPN网络，比如在RPN detection模型中采用的就是FPN neck
   从backbone获得的初始特征图是多层多尺度的，需要通过neck转换成同层多尺度，便于后续处理？？？？
    >输入：resnet的4个层特征[(2,256,152,256),(2,512,76,128),(2,1024,38,64),(2,2048,19,32)]
    >输出：list，包含5个元素[(2,256,152,256),(2,256,76,128),(2,256,38,64),(2,256,19,32),(2,256,10,16)]
    >计算过程：对每个特征图先进行1x1卷积，再进行递归上采样+相邻2层混叠，最后进行fpn3x3卷积抑制混叠效应
   FPN的优点：参考https://blog.csdn.net/baidu_30594023/article/details/82623623
    >同时利用了低层特征(语义信息少但位置信息更准确)和高层特征(语义信息更多但位置信息粗略)进行特征融合
     尤其是低层特征对小物体预测很有帮助
    >bottom-up就是模型前向计算，lateral就是横向卷积，top-down就是上采样+混叠+FPN3x3卷积
     如果只有横向抽取多尺度特征而没有进行邻层融合，效果不好作者认为是semantic gap不同层语义偏差较大导致
     如果只有纵向也就是只从最后一层反向上采样，效果不好作者认为是多次上采样导致特征位置更不准确

2. 方式2：通过backbone直接获得多尺度特征图，比如在SSD detection模型中就是直接从VGG16获得6路多尺度特征图

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# FPN的实现方式: 实现一个简单的FPN结构
class FPN_neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.outs_layers = 5                  # 定义需要FPN输出的特征层数
        self.lateral_conv = nn.ModuleList()   # 横向卷积1x1，用于调整层数为统一的256
        self.fpn_conv = nn.ModuleList()       # FPN卷积3x3，用于抑制中间上采样之后的两层混叠产生的混叠效应
        in_channels = [256,512,1024,2048]
        out_channels = 256
        for i in range(4):
            lc = nn.Sequential(nn.Conv2d(in_channels[i],out_channels, 1, 1, 0),  # 1x1保证尺寸不变，层数统一到256
                               nn.BatchNorm2d(out_channels),                                 # stride=1, padding=0
                               nn.ReLU(inplace=True))
            fc = nn.Sequential(nn.Conv2d(out_channels,out_channels, 3, 1, 1),    # 3x3保证尺寸不变，层数不变，注意s和p的参数修改来保证尺寸不变
                               nn.BatchNorm2d(out_channels),                                 # stride=1, padding=1
                               nn.ReLU(inplace=True))
            self.lateral_conv.append(lc)
            self.fpn_conv.append(fc)
            
    def forward(self, feats):
        lateral_outs = []
        for i, lc in enumerate(self.lateral_conv):
            lateral_outs.append(lc(feats[i]))       # 获得横向输出(4组)
        for i in range(2,0,-1):                     # 进行上采样和混叠(3组修改，1组不变)
            lateral_outs[i] += F.interpolate(lateral_outs[i+1], scale_factor=2, mode='nearest')
        outs = []
        for i in range(4):                          # 获得fpn卷积层的输出(4组)
            outs.append(self.fpn_conv[i](lateral_outs[i]))
        if len(outs) < self.outs_layers:            # 如果需要的输出特征图层超过当前backbone的输出，则用maxpool增加输出
            for i in range(self.outs_layers - len(outs)):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2)) # 用maxpool做进一步特征图下采样尺寸缩减h=(19-1+0)/2 +1=10
        return outs
  

# %% 
"""如何把特征图转化成提供给loss函数进行评估的固定大小的尺寸？
这部分工作一般是在head完成：为了确保输出的特征满足loss函数要求，需要根据分类回归的预测参数个数进行特征尺寸/通道调整
1. head的输入：
    >带FPN的head，输入特征图的通道数会被FPN统一成相同的(FPN做过上采样和叠加要求通道数一样)，比如RPN/FasterRCNN
    >不带FPN的head，输入通道数是变化的，比如ssd

2. head的输出：输出通道数就是预测参数个数，每一个通道(即一层)就用于预测一个参数
A.方式1：把分类和回归任务分开，分别预测，比如ssd/rpn/faster rcnn
    >分类支线，需要用卷积层预测bbox属于哪个类，所以需要的通道数n1 = 20
    >回归支线，需要用卷积层预测bbox坐标x/y/w/h，所以需要的通道数n2 = x,y,w,h = 4
    
B.方式2：把分类和回归任务一起做，一起预测，比如yolo
    >需要用卷积层同时预测bbox分类和bbox坐标，所以需要的通道数n=(x,y,w,h,c)+20类=25, 其中c为置信度
    
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
# 创建一个RPN head: 用于rpn/faster rcnn，
# 输入的特征来自FPN，通道数相同，所以分类回归只需要3个卷积层
class RPN_head(nn.Module):
    def __init__(self, num_anchors):
        super().__init__()
        self.num_anchors = num_anchors
        self.rpn_conv = nn.Conv2d(256, 256, 3, 1, 1)                   # conv1: 特征过滤
        self.rpn_cls = nn.Conv2d(256, self.num_anchors*2, 1, 1, 0)     # conv2: 多分类：输出通道数就是2*anchors，因为每个像素位置会有2个anchors (如果是2分类，也就是采用sigmoid而不是softmax，则只需要anchors个通道)
        self.rpn_reg = nn.Conv2d(256, self.num_anchors*4, 1, 1, 0)     # conv3: bbox回归：需要回归x,y,w,h四个参数
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
rpn = RPN_head(9)        # 每个数据网格点有9个anchors
results = rpn(fpn_outs)  # ([ft_cls0, ft_cls1, ft_cls2, ft_cls3, ft_cls4],
                         #  [ft_reg0, ft_reg1, ft_reg2, ft_reg3, ft_reg4])
    
# 创建一个SSD head: 用于ssd
# 输入的特征直接来自VGG，通道数不同，所以分类回归需要对应个数，比rpn/faster rcnn的卷积层要多一些
class SSD_head(nn.Module):
    def __init__(self):
        self.ssd_cls = []
        self.ssd_reg = []
        for i in range(6):
            self.ssd_cls.append(nn.Conv2d())
            
    def forward(self,feats):
        ft1 = self.rpn_conv(feats)
        ft2 = self.rpn_cls(F.relu(ft1))
        ft3 = self.rpn_reg(F.relu(ft1))
channels = [512,1024,512,256,256,256]
sizes = [(38,38),(19,19),(10,10),(5,5),(3,3),(1,1)]
        
# 创建一个YOLO head: 用于yolo
#



# %%    anchor的三部曲：(base anchor) -> (anchor list) -> (anchor target)
"""Q.如何产生base anchors?
base anchors是指特征图上每个网格上的anchor集合，通常一个网格上会有3-9个anchors
1. 为什么是9个base anchor, 为什么是这些尺寸的anchors?
    在yolov3的文献中介绍说这些anchor是通过对数据集中所有anchors进行聚类，聚类参数k从1-15, 
    (k再大则模型复杂度太高了，因为anchor过多)发现k=9所得平均重合IOU最优，虽然每次聚类的9个anchor稍有不同，
    但基本上在voc上平均IOU大约是67%左右(可参考_test_yolov3_kmeans_on_voc.py)
    而从实际的base anchor大小来看，这个尺寸也可以这么理解：首先base anchor大小定义成正好覆盖
    一个特征图网格所对应的原始图size的区域，也就是stride x stride的大小，这是符合逻辑的第一步思考，
    然后通过scale定义，最小的anchor是(8 x stride x stride)的正方形,这个8倍/16倍/32倍
    应该是基于实际图片上的最小/最大bbox尺寸来定义出来的，确保所有小物体框都能被包含，这也跟yolo所谓的
    聚类是一个效果。在rpn detection中最小anchor是8x4x4,也就是128x128像素的框，这非常小了，gt_bbox应该没有比这更小
2. 要生成每一层特征层需要的anchors，分成如下几步：
    >定义anchor base size：这步定义的base size=[4,8,16,32,64]取决于特征图的size大小，
     比如256边界的anchor size取4，128边界取8，64边界取16, 
    >创建base anchors
3. 对每层特征图使用一定数量anchors：
    >比如rpn/fasterrcnn针对每层特征图使用9个base anchors, 而yolo针对每层特征图使用3个base anchors，
    >原则是越大的特征图，则使用越小的anchors: 因为越大特征图就是越浅层特征图，
     因此特征趋向于低语义/小尺寸物体，适合小尺寸anchors
"""
def gen_base_anchors_mine(anchor_base, ratios, scales, scale_major=True):
    """生成n个base anchors/[xmin,ymin,xmax,ymax],生成的base anchors的个数取决于输入
    的scales/ratios的个数，早期一般输入3个scale和3个ratio,则每个网格包含9个base anchors
    现在一些算法为了减少计算量往往只输入一个scale=8, 而ratios输入3个[0.5, 1.0, 2.0]，
    所以对每个网格就包含3个base anchors
    生成的base anchor大小取决与anchor base大小，由于每个特征图的anchor base都不同，
    所以每个特征图对应base anchor大小也不同，浅层大特征图由于stride小，对应anchor base
    也小，也就是大特征图反而对应小anchor，数量更多的小anchor
    Args:
        anchor_base(float): 表示anchor的基础尺寸
        ratios(list(float)): 表示h/w，由于r=h/w, 所以可令h'=sqrt(r), w'=1/sqrt(r), h/w就可以等于r了
        scales(list(float)): 表示整体缩放倍数
        scale_major(bool): 表示是否以scale作为anchor变化主体，如果是则先乘scale再乘ratio
    Returns:
        base_anchors(tensor): (m,4)
    1. 计算h, w和anchor中心点坐标(是相对于图像左上角的(0,0)点的相对坐标，也就是假设anchor都是在图像左上角
       后续再通过平移移动到整个图像每一个网格点)
        h = base * scale * sqrt(ratio)
        w = base * scale * sqrt(1/ratio)
        x_ctr = h/2
        y_ctr = w/2
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
    
import torch    
anchor_strides = [4., 8., 16., 32., 64.]
anchor_base_sizes = anchor_strides      # 基础尺寸
anchor_scales = [8., 16., 32.]          # 缩放比例
anchor_ratios = [0.5, 1.0, 2.0]         # w/h比例

num_anchors = len(anchor_scales) * len(anchor_ratios)
base_anchors = []
for anchor_base in anchor_base_sizes:
    base_anchors.append(gen_base_anchors_mine(anchor_base, anchor_ratios, anchor_scales))
    

# %%    anchor的三部曲：(base anchor) -> (anchor list) -> (anchor target)
"""Q.如何产生anchor list?
anchor list产生目的就是要让feature map上每一个网格所对应的原图区域都放置n个base anchors
这个base anchors是根据stride步幅，以及自定义的anchor scale/ratio生成的。
比如stride步幅是4, 那么base size定义就是4x4(base size的大小很小，只是正好覆盖原图对应区域)，
然后缩放[8,16,32],调整h/w比例[0.5,1,2]得到实际anchor大小(实际anchor大小远超对应区域，至少是8倍base size大小)
最后平移这组anchor到原图的每一块对应区域(对应取余的面积就是stride x stride)
"""
def grid_anchors_mine(featmap_size, stride, base_anchors):
    """基于base anchors把特征图每个网格所对应的原图感受野都放置base anchors
    Args:
        featmap_size(list(float)): (a,b)
        stride(float): 代表该特征图相对于原图的下采样比例，也就代表每个网格的感受野
                      是多少尺寸的原图网格，比如1个就相当与stride x stride大小的一片原图
        device(str)
    Return:
        all_anchors(tensor): (n,4), 这里的n就等于特征图网格个数*每个网格的base anchor个数(比如3或9个)
    1. 先计算该特征图对应原图像素大小 = 特征图大小 x 下采样比例
    2. 然后生成网格坐标xx, yy并展平：先得到x, y坐标，再meshgrid思想得到网格xx, yy坐标，再展平
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
    
    shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1) # 堆叠成4列给4个坐标[xmin,ymin,xmax,ymax], (38912,4)
    shifts = shifts.type_as(base_anchors)   # 从int64转换成torch认可的float32
    
    # anchor坐标(9,4)需要基于网格坐标(38912,4)进行平移：平移后应该是每个网格点有9个anchor
    # 也就是38912个(9,4)，也就等效于anchor(9,4)与每一个网格坐标(1,4)进行相加
    # 需要想到把(38912,4)提取出(1,4)的方式是升维到(38912,1,4)与(9,4)相加
    all_anchors = base_anchors + shifts[:,None,:]   # 利用广播法则(9,4)+(38912,1,4)->(39812,9,4)
    all_anchors = all_anchors.view(-1,4)            # 部分展平到(n,4)得到每个anchors的实际坐标(图像左上角为(0,0)原点)                      
    
    return all_anchors

import torch    
anchor_strides = [4., 8., 16., 32., 64.]
anchor_base_sizes = anchor_strides      # 用stride作为anchor的base size: 道理在于网格划分也是按stride划分，则在原图的stride x stride区域正好可以被一个base size的anchor覆盖，确保这个网格中的物体能够被检测出来
anchor_scales = [8., 16., 32.]          # 缩放比例
anchor_ratios = [0.5, 1.0, 2.0]         # w/h比例

num_anchors = len(anchor_scales) * len(anchor_ratios)
base_anchors = []
for anchor_base in anchor_base_sizes:   # 先计算得到base anchors
    base_anchors.append(gen_base_anchors_mine(anchor_base, anchor_ratios, anchor_scales))

featmap_sizes = [(152,256), (76,128), (38,64), (19,32), (10,16)]
strides = [4,8,16,32,64]    # 针对resnet的下采样比例，5路分别缩减尺寸 

i=0 # 取第0个特征图的相关参数：特征图尺寸，步幅(也就是下采样比例)，事先生成的base anchors
featmap_size = featmap_sizes[i]
stride = strides[i]
base_anchor = base_anchors[i]
all_anchors2 = grid_anchors_mine(featmap_size, stride, base_anchor)


# %%
"""Q. 如何对all anchors进行筛选，一个核心方法是IOU，那如何进行IOU计算？
"""
def bbox_overlap_mine(bb1, bb2, mode='iou'):
    """bbox的重叠iou计算：iou = Intersection-over-Union交并比(假定bb1为gt_bboxes)
       还有一个iof = intersection over foreground就是交集跟gt_bbox的比
    Args:
        bb1(tensor): (m, 4) [xmin,ymin,xmax,ymax], 通常取bb1输入gt
        bb2(tensor): (n, 4) [xmin,ymin,xmax,ymax], 通常取bb2输入all anchors
    Return:
        ious(tensor): (m,n) 代表的就是m行gt跟n列anchors的ious网格
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
    wh = (xymax -xymin).clamp(min=0)   # 得到宽高w, h (m,n,2)
    
    overlap = wh[:,:,0] * wh[:,:,1]   # (m,n)*(m,n) -> (m,n),其中m个gt的n列w, 乘以m个gt的n列h
    
    ious = overlap / (area1[:, None] + area2 -overlap) # 由于m个gt的每一个面积都要跟n的每一个面积相加，要得到(m,n)的面积之和
                                                       # 所以需要升维(m,1)+(n)->(m,n), 然后(m,n)-(m,n)，以及(m,n)/(m,n)都可以操作
    return ious

bb1 = torch.tensor([[-20.,-20.,20.,20.],[-30.,-30.,30.,30.]])
bb2 = torch.tensor([[-25.,-25.,25.,25.],[-15.,-15.,15.,15.],[-25,-25,50,50]])
ious2 = bbox_overlap_mine(bb1, bb2)


# %%    anchor的三部曲：(base anchor) -> (anchor list) -> (anchor target)
"""Q.如何对all anchor进行指定与标记?
对anchor进行标记的目的是对anchor进行初步筛选，确保每个anchor要么是正样本(1)要么是负样本(0)要么是无关样本(-1)
其中负样本是指与gt的0<iou<0.3, 正样本是指与gt的iou>0.7以及等于该gt对应最大iou的anchors, 无关样本是指剩余样本
是通过assigner()指定器来完成，最终输出一组对每个anchor的身份指定的tensor
"""
def assigner(bboxes, gt_bboxes):
    """anchor指定器：用于区分anchor的身份是正样本还是负样本还是无关样本
    正样本标记为1+n(n为第几个gt), 负样本标记为0, 无关样本标记为-1
    Args:
        bboxes(tensor): (m,4)
        gt_bboxes(tensor): (n,4)
    Return:
        assigned(tensor): (m,) 代表m个bboxes的身份tensor，其值value=[-1,1,2..n]分别表示所对应的gt(-1表示无关，1~n表示第1~n个gt，没有0)
    1. 先创建空矩阵，值设为-1
    2. 再把所有0<iou<0.3的都筛为负样本(0)，iou>0.7的都筛为正样本(1+idx)
    3. 再把该gt最适配的anchor也标为正样本(1+idx)：即gt对应的iou最大的anchor
       注意基于gt找到的iou最高的anchor，往往不是该anchor的最高iou，所以这一步是把anchor中只要高于该iou的所有anchor都提取为fg
    
    """
    pos_iou_thr = 0.7  # 正样本阀值：iou > 0.7 就为正样本
    neg_iou_thr = 0.3  # 负样本阀值：iou < 0.3 就为负样本
    min_pos_iou = 0.3  # 预测值最小iou阀值，确保至少这个阀值应该大于负样本的最大阀值
    overlaps = bbox_overlap_mine(gt_bboxes, bboxes) # (m,n)代表m个gt, n个anchors
    n_gt, n_bbox = overlaps.shape
    # 第一步：先创建一个与所有anchor对应的矩阵，取值-1(代表没有用的anchor)
    assigned = overlaps.new_full((overlaps.size(1),), -1, dtype=torch.int64)  # (n,)对应n个anchors, 填充-1表示无关样本
                                                                              # 注意这里dtype要改一下，否则跟下面相加的int64冲突
    max_overlap, argmax_overlap = overlaps.max(dim=0)      # (n,)对应n个anchors，表示每个anchor跟哪一个gt的iou最大 (该变量跟assigned同尺寸，用来给assigned做筛选)
    gt_max_overlap, gt_argmax_overlap = overlaps.max(dim=1)# (m,)对应m个gt，表示每个gt跟那个anchor的iou最大
    # 第二步：标记负样本，阀值定义要经可能让负样本数量跟正样本数量相当，避免样本不平衡问题
    assigned[(max_overlap >= 0) & (max_overlap < neg_iou_thr)] = 0  # 0< iou <0.3, value=0
    # 第三步：标记正样本，阀值定义要经可能让负样本数量跟正样本数量相当，避免样本不平衡问题
    # 注意：value = 1 + n, 其中n为第n个gt的意思，所以value范围[1, n_gt+1], value值正好反映了所对应的gt
    assigned[max_overlap >= pos_iou_thr] = 1 + argmax_overlap[max_overlap >= pos_iou_thr] # iou >0.7, value = 1 + 位置值
    # 第四步：标记预测值foreground(也称前景)，也就是每个gt所对应的最大iou为阀值, 但这个阀值先要判断至少大于负样本你的上阀值
    # 注意：只要取值等于该gt的最大iou都被提取，通常不止一个最大iou。value值范围[1,n_gt+1]代表所对应gt
    for i in range(n_gt):
        if gt_max_overlap[i] >= min_pos_iou:   #如果gt最适配的anchor对应iou大于阀值才提取
            max_iou_idx = overlaps[i]==gt_max_overlap[i] # 从第i行提取iou最大的位置的bool list
            assigned[max_iou_idx] = 1 + i   # fg的value比正样本的value偏小
    return assigned

import pickle
gt_bboxes = pickle.load(open('test/test_data/test9_bboxes.txt','rb'))  # gt bbox (m,4)
gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)
bboxes = all_anchors2  # (n,4)
ious = bbox_overlap_mine(gt_bboxes, bboxes)

assign_result = assigner(bboxes, gt_bboxes)


# %%
"""Q. 如何基于已指定身份的anchor进行采样提取？
对anchor进行采样的目的：从all anchors里边挑选出256个正负样本，其中正负样本数量基本接近
"""
def random_sampler(assigned, bboxes):
    """anchor抽样器: 基于随机采样方式,从all anchors里边先分离出正样本和负样本，
    然后在正负样本中分别按照比例抽取总数固定的样本个数用于训练(通常抽需256个样本)
    Args:
        assigned(tensor): (m,) 代表m个anchor的身份指定, 取值范围[-1,1,2,..n]
        bboxes()
    Return:
        pos_inds(tensor): (j,) 代表指定数量的anchor正样本的index列表
        neg_inds(tensor): (k,) 代表指定数量的anchor负样本的index列表
        
    """
    num_expected = 256   # 总的采样个数
    pos_fraction = 0.5   # 正样本占比
    num_pos = int(num_expected * pos_fraction)
    # 正样本抽样：通常正样本数量较少，不会超过num ecpected
    pos_inds = torch.nonzero(assigned > 0)  # (j,1) 正样本的index号
    if torch.numel(pos_inds)!=0:
        pos_inds = pos_inds.squeeze(1)      # (j,)
    if torch.numel(pos_inds) > num_pos:  # 如果正样本数太多则抽样
        pos_rand_inds = torch.randperm(len(pos_inds))[:num_pos] # 先对index随机排序，然后抽取前n个数
        pos_inds = pos_inds[pos_rand_inds]
#        candidates = np.arrange(len(pos_inds))  # 也可用numpy来实现采样，速度比torch快
#        np.random.shuffle(candidates)
#        rand_inds = cnadidates[:num_expected*pos_fraction]
#        return pos_inds[]
    # 负样本抽样：通常负样本数量较多，所有anchors里边可能70%以上iou都>0，即都为负样本
    neg_inds = torch.nonzero(assigned == 0) # (k, 1)负样本的index号
    if torch.numel(neg_inds)!=0:
        neg_inds = neg_inds.squeeze(1)      # (k,)
    if torch.numel(neg_inds) > num_expected - num_pos: # 如果负样本数太多则抽样
        neg_rand_inds = torch.randperm(len(neg_inds))[:(num_expected - num_pos)]
        neg_inds = neg_inds[neg_rand_inds]
    return pos_inds, neg_inds

import pickle
gt_bboxes = pickle.load(open('test/test_data/test9_bboxes.txt','rb'))  # gt bbox (m,4)
gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)
bboxes = all_anchors2  # (n,4)
ious = bbox_overlap_mine(gt_bboxes, bboxes)

assign_result = assigner(bboxes, gt_bboxes)

sample_result = random_sampler(assign_result, bboxes)


# %%
"""Q. 从assigner/sampler得到正样本的anchors后，如何跟gt bbox进行修正，让proposal更接近gt？
从assigner/sampler得到正负样本的ind，转换后就能得到正负样本anchor的坐标，称为proposals(j,4)
为了跟实际gt bbox进行比较，只需要提取正样本的bbox坐标组，以及该正样本所对应的gt bbox的坐标组
两者进行bbox回归，g = f(p)，p为proposal anchors，g为gt bbox，然后求出f函数的参数dx,dy,dw,dh
每一组anchor对应了一组(dx,dy,dw,dh)
"""
def bbox2delta():
    """"""
    pass

def delta2bbox():
    """"""
    pass
    

# %%
"""Q. 从assigner/sampler得到正样本的anchors后，最终如何生成anchor targets?

"""
def anchor_target_mine(gt_bboxes, inside_anchors, inside_f, assigned, 
                       pos_inds, neg_inds, num_all_anchors, num_level_anchors, gt_labels):
    """anchor目标：首先对anchor的合法性进行过滤，取出合法anchors(没有超边界)，
    注意，这里的inside_anchors需要是经valid_flag/inside_flag过滤的anchors，
    同时传入assigner/sampler的也应该是valid_flag/inside_flag过滤的anchors得到的输出assigned/pos_inds/neg_inds
    首先生成正样本的回归参数(dx,dy,dw,dh)，然后生成对应的正样本权重=1，
    再生成正样本标签=1，和正样本+负样本标签权重=1
    再借用inside_flags把求得的target/weights都映射回原始all_anchors的尺寸中。
    最后借用num_level_anchors把求得的target/weights都按anchors的分布数分割到每一个level
    注1：对于多尺度anchor的处理方式就是在anchor target之前把多尺度特征的anchor list/valid flag
    先concatenate在一列，就相当与单尺度问题了。而对于batch的多图片问题，则需要multi_apply()解决
    注2：为了保证loss的输入格式，这里把输出额外加了一个list[]，后续改为多图时，用multi_apply也会放在list里
    Args:
        gt_bboxes(tensor): (m,4) 代表标签bboxes
        inside_anchors(tensor): (n,4) 代表网格上的anchors在图像边沿以内的anchors(即在inside_flags中的anchors)
        inside_f(tensor): (n_max,) 代表筛选所有anchors在图像边沿以内的标签
        assigned(tensor): (n,) 指定器输出结果，代表n个anchor的身份指定[-1,0,1,2..m]
        pos_inds(tensor): (j,) 采样器输出结果，代表j个采样得到的正样本anchors的index
        neg_inds(tensor): (k,) 采样器输出结果，代表k个采样得到的负样本anchors的index
        num_all_anchors(int): (n_max,) 初始生成的所有all_anchors(没有经过valid_flag/inside_flag过滤)
        gt_labels(tensor): (m,) optional可输入gt对应标签，也可不输入
    Return:
        labels_list([tensor]): (n_max,) 代表的是正样本所在位置的标签，默认取1, 非正样本取0
        labels_weights_list([tensor]): (n_max, ) 代表的是正样本+负样本所在位置的权重，默认取1，其他无关样本取0
        bbox_targets_list([tensor]): (n_max,4) 代表的是正样本所对应的回归函数参数(dx,dy,dw,dh), 非正样本为0
        bbox_weights_list([tensor]): (n_max,4) 代表对应正样本所对应参数坐标的权重(1,1,1,1), 非正样本为0
    """
    
    # 先创建0数组
    bbox_targets = torch.zeros_like(inside_anchors)  # (n,4)
    bbox_weights = torch.zeros_like(inside_anchors)  # (n,4)
    labels = inside_anchors.new_zeros(inside_anchors.shape[0],dtype=torch.int64) # (n,)
    labels_weights = inside_anchors.new_zeros(inside_anchors.shape[0], dtype= torch.float32) # (n,)
    # 采样index转换为bbox坐标
    pos_bboxes = inside_anchors[pos_inds]  # (j,4)正样本index转换为bbox坐标
    # 生成每个正样本所对应的gt坐标，用来做bbox回归
    pos_assigned = assigned[pos_inds] - 1       # 提取每个正样本所对应的gt(由于gt是大于1的1,2..)，值减1正好就是从0开始第i个gt的含义
    pos_gt_bboxes = gt_bboxes[pos_assigned,:]   # (j,4) 生成每个正样本所对应gt的坐标
    if len(pos_inds) > 0:
        #对正样本相对于gt做bbox回归
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes) # (j, 4)得到的是每个proposal anchor对应回归target的回归参数
        # 更新bbox_targets/bbox_weights
        bbox_targets[pos_inds, :] = pos_bbox_targets  # 所有anchor中正样本的坐标更新为targets的deltas坐标
        bbox_weights[pos_inds, :] = 1.0               # 所有anchor中正样本的权重更新为1
        # 更新labels/labels_weights
        labels[pos_inds] = 1            # 默认gt_labels=None，所以labels对应target的位置设置为1
        labels_weights[pos_inds] = 1.0  # cfg中pos_weight可自定义，如果定义-1说明用默认值则设为1
    if len(neg_inds) > 0:
        labels_weights[neg_inds] = 1.0

    # unmap: 采用默认的unmap_outputs =True
    # unmap的目的是把inside_anchors所对应的输出映射回原来all_anchors
    # 这里做了额外处理：由于只是验证单图，没有multi_apply，也就没有把tensor装在list中，所以手动加了list外框
    labels = [unmap(labels, num_all_anchors, inside_f)]
    labels_weights = [unmap(labels_weights, num_all_anchors, inside_f)]
    bbox_targets = [unmap(bbox_targets, num_all_anchors, inside_f)]
    bbox_weights = [unmap(bbox_weights, num_all_anchors, inside_f)]
    
    # 计算total_pos/total_neg: 单图和为256, 如果多图batch则需要累加
    num_total_pos = len(pos_inds)
    num_total_neg = len(neg_inds)
    
    # 分解到每一个特征图层上：
    labels_list = distribute_to_level(labels, num_level_anchors)
    labels_weights_list = distribute_to_level(labels_weights, num_level_anchors)
    bbox_targets_list = distribute_to_level(bbox_targets, num_level_anchors)
    bbox_weights_list = distribute_to_level(bbox_weights, num_level_anchors)
    
    return (labels_list, labels_weights_list, 
            bbox_targets_list, bbox_weights_list,
            num_total_pos, num_total_neg)

def unmap(data, total, inds, fill=0):
    """借用inside_flags把得到的data映射回原来的total数据中：
    即创建一个跟原来all anchors尺寸一样的0数组，然后把target数据放入指定位置
    """
    if data.dim() == 1:
        unmapped = data.new_full((total,), fill)
        unmapped[inds] = data
    else:
        new_size = (total, ) + data.size()[1:]       # (m,) + (n,) = (m,n)  
        unmapped = data.new_full(new_size, fill)
        unmapped[inds, :] = data
    return unmapped

def distribute_to_level(all_data, num_level_anchors):
    """借用每个特征图上产生的anchors个数，把所有anchor target的生成变量
    按照anchors的数量分布，都分配到每个特征图上去
    Args:
        all_data(list): [tensor1_img1, tensor2_img2...] 代表每张图片对应的targets输出变量(可以是bbox_targets/weights/labels..)
        num_level_anchors(list): [num_level1, num_level,..] 代表每级特征图上anchors的个数,比如[105792, 26448, 6612, 1653, 450]
    Returns:
        level_data(list): [level1, level2,...] 代表每个level包含的数据tensor, 如果是多图，则tensor为多行来表示，单图则tensor为单行
    """
    all_data = torch.stack(all_data, 0)  # 列方向堆叠，把多图多tensor堆叠成一个tensor -> (n_img, n_anchors)
    level_data = []
    start = 0
    for n in num_level_anchors:
        end = start + n 
        level_data.append(all_data[:,start:end].squeeze(0))  # 按level分割
        start = end
    return level_data



# %%
"""Q. 得到的anchor targets到底怎么理解，怎么做loss损失计算？
"""
def loss_single(cls_score, bbox_pred, labels, labels_weights, 
                bbox_targets, bbox_weights, num_total_samples):
    """基于卷积网络的输出和anchor target的输出，进行单张图片的损失计算
    Args:
        cls_score(tensor): (b,c,h,w) 代表head最终输出的特征图比如(2,3,152,240)
        bbox_pred(tensor): (b,x,h,w) 其中x是由预测bbox的尺寸个数决定的(a个预测框*b个预测框参数，比如=3*(xmin,ymin,xmax,ymax)=3*4=12)
                            比如(2,12,152,240)
        labels(tensor): (n_max,) 代表的是正样本所在位置的标签，默认取1, 非正样本取0
        labels_weights(tensor): (n_max, ) 代表的是正样本+负样本所在位置的权重，默认取1，其他无关样本取0
        bbox_targets(tensor): (n_max,4) 代表的是正样本所对应的回归函数参数(dx,dy,dw,dh), 非正样本为0
        bbox_weights(tensor): (n_max,4) 代表对应正样本所对应参数坐标的权重(1,1,1,1), 非正样本为0
        num_total_samples(int): 代表？？
    Return:
        loss_cls(tensor): (1,)  分类损失
        loss_reg(tensor): (1,)  回归损失
    """
    # 分类损失计算    
    labels = labels.reshape(-1,1)
    labels_weights = labels_weights.reshape(-1,1)
    cls_score = cls_score.permute(0,2,3,1).reshape(-1,1) # (b,c,h,w)->(b,h,w,c)->(x,)
    cls_criterion = weighted_binary_cross_entropy
    loss_cls = cls_criterion(cls_score, labels, labels_weights, 
                             avg_factor=num_total_samples)
    # 回归损失计算
    bbox_targets = bbox_targets.reshape(-1,4)
    
    loss_reg = weighted_smoothl1(bbox_pred, bbox_targets, bbox_weights,
                                 beta = 1./9., avg_factor=num_total_samples)
    return loss_cls, loss_reg




# %%
"""Q. 如何对物体检测卷积网络的输出进行损失评估，跟常规分类网络的评估有什么区别？
1. 分类损失
2. 回归损失
"""


# %% 
"""常说的One stage和two stage detection的主要区别在哪里？
"""


# %% 
"""新出的RetinaNet号称结合了one-stage, two-stage的优缺点，提出的Focal loss有什么特点？
是否采用Focal loss就不需要对不平衡数据进行sampling采样？那Focal loss如何解决数据不平衡问题？
"""


# %% 
"""Q. 对已经训练完成的模型进行测试，第一步模型加载如何做？"""
# model_zoo
import os
import OrderedDict
def load_url(url, model_dir=None, map_location=None, progress=True):
    """pytorch用来在线下载模型以及从torch home打开checkpoint
    如下代码做了删减，只保留了从本地torch home调用已下载好的模型，但模型地址可以写http，也可以写目录
    """
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)   
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    
    return torch.load(cached_file, map_location=map_location)
# 测试一下，如下是我已经从mmdetection下载好放在.torch的torch home的一个模型
url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_fpn_1x_20181010-4a9c0712.pth'

# checkpoint
def load_checkpoint_mine(model, filename, map_location=None, strict=False, logger=None):
    """mmdetection用来加载模型参数的代码, 也做了修改，去掉了
    """
    if filename.startswith('modelzoo://'):
        from torchvision.models.resnet import model_urls
        model_name = filename[11:]
        checkpoint = load_url(model_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url(filename)
    else:
        if not os.path.isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))
        checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint

# %% 
"""Q. 在测试模式下，模型如何计算？
"""





# %%
"""Q. 如何对model效果进行test和评估？
注意1：由于所有mmdetection所提供的模型都是基于coco进行训练的，所以测试所搭建的模型需要
采用针对coco数据集的模型和cfg文件
0. 可以test一张或几张图片，也可以test整个数据集的test数据
    几张张图片的测试调用api中的：result = inference_detector() -> _inference_single()/_inference_generator()
                          show_result(img, result)
    数据集的测试调用：
"""
import torch
import cv2
import mmcv
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.datasets.transforms import ImageTransform
from mmdet.core import get_classes
import numpy as np
from B03_dataset_transform import imshow_bboxes_labels 

def test_img(img_path, config_file, device = 'cuda:0', dataset='coco'):
    """测试单张图片：相当于恢复模型和参数后进行单次前向计算得到结果
    注意由于没有dataloader，所以送入model的数据需要手动合成img_meta
    1. 模型输入data的结构：需要手动配出来
    2. 模型输出result的结构：
    Args:
        img(array): (h,w,c)-bgr
        config_file(str): config文件路径
        device(str): 'cpu'/'cuda:0'/'cuda:1'
        dataset(str): voc/coco
    """
    # 1. 配置文件
    cfg = mmcv.Config.fromfile(config_file)
    cfg.model.pretrained = None
    # 2. 模型
    path = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
    model = build_detector(cfg.model, test_cfg = cfg.test_cfg)
    _ = load_checkpoint(model, path)
    model = model.to(device)
    # 3. 图形/数据变换
    img = cv2.imread(img_path)
    img_transform = ImageTransform(size_divisor = cfg.data.test.size_divisor,
                                   **cfg.img_norm_cfg)
    # 5. 数据包准备
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(img, scale= cfg.data.test.img_scale)
    img = torch.tensor(img).to(device).unsqueeze(0)
    
    img_meta = [dict(ori_shape=ori_shape,
                     img_shape=img_shape,
                     pad_shape=pad_shape,
                     scale_factor = scale_factor,
                     flip=False)]

    data = dict(img=[img], img_meta=[img_meta])
    # 6. 结果计算
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    # 7. 结果显示
    class_names = get_classes(dataset)
    labels = [np.full(bbox.shape[0], i, dtype=np.int32) 
        for i, bbox in enumerate(result)]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    img = cv2.imread(img_path)
    
#    imshow_bboxes_labels(img.copy(), bboxes, labels, 
#                         score_thr=0.7,
#                         class_names=class_names,
#                         bbox_colors=(0,255,0),
#                         text_colors=(0,255,0),
#                         thickness=1,
#                         font_scale=0.5)
    
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        bbox_color='blue',
        text_color='blue',
        class_names=class_names,
        score_thr=0.5)
    
img_path = 'test/test_data/001000.jpg'    
config_file = 'test/test_data/cfg_fasterrcnn_r50_fpn_coco.py'
test_img(img_path, config_file)


# %%
"""Q. 如何对整个数据集进行测试评估？

"""
def test_dataset(config_file, checkpoint_file, gpus, out_file, eval_method='proposal_fast'):
    """测试一个数据集
    参考mmdetection/tools/test.py
    命令行用法：python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --gpus <GPU_NUM> --out <OUT_FILE>
    执行过程：
    1. 调用
    2. 调用
    Args:
        config_file(str): 代表测试配置文件的路径(.py)，通常跟训练配置文件集成在一起的cfg
        checkpoint_file(str): 代表模型文件的路径(.pth)
        gpus(int): 代表gpus的个数(1~n)
        out_file(str): 代表输出文件地址和文件名(.pkl)
        eval_method(str): 代表评估方式, proposal_fast则表示
    """
    from mmcv.runner import obj_from_dict
    from mmdet import datasets
    
    cfg = mmcv.Config.fromfile(config_file)
    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    
    if args.gpus == 1:
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint)
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader, args.show)
    else:
        raise ValueError('currently only support one gpu for test dataset.')
    
    checkpoint_file
    gpus
    out_file


# %%
"""Q. 其他对检测性能帮助较大的方法？
参考：亚马逊的李沐的文章《bag of freebies for training object detection neural networks》-2019-2-11
主要观点包括：
1. visually coherent image mixup for object detection:
    就是对不同图像进行mixup，比如0.9*img1 + 0.1*img2，混合比例对map影响较大，需要调参
2. classification head label smoothing:
    就是对head的label进行smooth化，这个是否在mmdetection有体现需要看一下
3. data pre-processing:
    就是数据增广，比如随机几何变换(随机裁剪，随机水平翻转，随机缩放)，随机颜色抖动(亮度，色调，饱和度，对比度)
4. traning scheduler revamping:
    就是学习率的优化，包括warm up lr, step schedule..
5. synchronized batch normalization:
    SBN
6. random shapes traning for single stage object detection networks:
    由于single stage模型使用固定形状，如果采用对图像h,w进行随机变换，可以提高最多5个点
以上代码都已开源在gluon，有必要学习下。
"""


# %%
"""总体验证"""
def main(step=1):
#--------验证FPN的输出------------
    if step == 1:
        channels = [256,512,1024,2048]
        sizes = [(152,256),(76,128),(38,64),(19,32)]
        # 构造假数据
        feats = []
        for i in range(4):  
            feats.append(torch.randn(2, channels[i], sizes[i][0], sizes[i][1]))
        # 计算输出
        fpn = FPN_neck()
        outs = fpn(feats)  # 输出5组特征图(size为阶梯状分辨率，152x256, 76x128, 38x64, 19x32, 10x16)
                           # 该输出特征size跟输入图片尺寸相关，此处256-128-64-32-16是边界
#--------验证FPN的输出------------
    if step == 2:
        pass
    
if __name__ == '__main__':
    main()


