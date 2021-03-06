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
2. 物体检测/object detectin：需要识别具体物体的类型和位置，比如：这张图形有7个气球，每个气球的位置框
3. 语义分割/Semantic segmentation：需要识别气球，还要把气球从背景中整体分离出来，比如：这张图片有7个气球，分成气球和背景两部分
4. 实例分割/Instance Segmentation：需要识别气球，还要吧每个气球都单独分离并给出每个气球像素，比如：这张图片有7个气球，分成7组独立气球和1组背景
可以认为从1到4难度逐渐加大。
参考：shalock的物体检测综述(对这4块有明确的定位)
参考：https://blog.csdn.net/qq_15969343/article/details/80167215
"""

# %%
"""Q. 如何区分对象检测的one stage和two stage?
1. two stage:
    > 两个阶段是指两个head，分别对feat进行两轮loss计算
    > (2014)RCNN是第一个提出region的概念，(2015)后续fast rcnn/faster rcnn都是如何更快找到region
    > (2016)R-FCN在faster rcnn基础上又有了较大提高(代继峰)
    > (2017)FPN提出了多尺度特征金字塔网络，而mask rcnn则在faster rcnn基础上增加mask branch(head)用来做实例分割
      且由于多任务学习，他对物体框的性能也有很大提高
    > (2018)Cascade rcnn把cascade结构用在faster rcnn，结合在不同stage对iou阈值的调整，性能获得很大提高
    > 阶段1的RPN head重点在anchor，目的是获得proposals，虽然也有使用分类和回归的数据，但主要是为了产生合适的proposal并不是真的做结果的分类和回归。
    > 阶段2的Bbox head重点在bbox，目的是进行最终结果的分类和回归

2. One stage:
    > (2014)最早有multibox的概念
    > (2016)产生SSD和Yolo
    > (2017)产生RetinaNet和yolo v2
    > (2018)产生CornerNet

"""

            
# %%
'''Q. Resnet作为物体检测的backbones跟常规有什么区别，如何提取多尺度特征图？
'''    
# resnet作为backbone的修改




# %%
"""Q. VGG16作为物体检测的backbones跟常规有什么区别，如何提取多尺度特征图？"""
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
    def __init__(self, num_outs = 5):
        super().__init__()
        self.outs_layers = num_outs                  # 定义需要FPN输出的特征层数
        self.lateral_conv = nn.ModuleList()   # 横向卷积1x1，用于调整层数为统一的256
        self.fpn_conv = nn.ModuleList()       # FPN卷积3x3，用于抑制中间上采样之后的两层混叠产生的混叠效应
        in_channels = [256,512,1024,2048]     # 配置每层输入输出
        out_channels = 256                    # FPN输出特点：层数相同
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


"""验证说明：输入4个特征图尺寸[[2,256,152,256],[2,512,76,128],[2,1024,38,64],[2,2048,19,32]]
经过FPN后变为5个输出特征图尺寸[[2,256,152,256],[2,256,76,128],[2,256,38,64],[2,256,19,32],[2,256,10,16]]
"""
channels = [256,512,1024,2048]                 # 构造4层特征图 - 层数 
sizes = [(152,256),(76,128),(38,64),(19,32)]   # 构造4层特征图 - 尺寸
feats = []
for i in range(4):                             # 构造4层特征图 - 数据
    feats.append(torch.randn(2, channels[i], sizes[i][0], sizes[i][1]))
fpn = FPN_neck(num_outs=5)
fpn_outs = fpn(feats)    # 输出


# %% 
"""如何把特征图转化成提供给loss函数进行评估的固定大小的尺寸？
这部分工作可以在rpn head完成：为了确保输出的特征满足loss函数要求，需要根据分类回归的预测参数个数进行特征尺寸/通道调整
1. rpn head的输入：
    >是已经被FPN统一成相同层数的多层特征图(5,)，例如(t1,t2,t3,t4,t5)
2. rpn head的输出：本质就是调整层数，基于每个cell的anchor个数n_a
   所以分解成2组(2,5)，例如[[cls1,cls2,cls3,cls4,cls5],[reg1,reg2,reg3,reg4,reg5]]
   (以下调整层数的公式，na代表每个cell的anchors个数，out代表实际输出通道个数)
    >cls: (b, c, h, w) -> (b, out*na, h, w), 其中out*na代表输出二分类问题输出通道应该为2(即2*na), 但如果是用sigmoid()则取out=1，用softmax则取out=2
    >reg: (b, c, h, w) -> (b, 4*na, h, w), 其中4*na代表了4倍anchor个数，每层代表一个anchor回归参数(dx,dy,dw,dh)

3. 对比不同head
A.方式1：把分类和回归任务分开，分别预测，比如ssd/rpn/faster rcnn
    >分类支线，需要用卷积层预测bbox属于哪个类，所以需要的通道数n1 = 20
    >回归支线，需要用卷积层预测bbox坐标x/y/w/h，所以需要的通道数n2 = x,y,w,h = 4
    
B.方式2：把分类和回归任务一起做，一起预测，比如yolo
    >需要用卷积层同时预测bbox分类和bbox坐标，所以需要的通道数n=(x,y,w,h,c)+20类=25, 其中c为置信度
    
"""
import torch
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
"""验证说明：首先需要运行上个cell得到fpn_outs [[2,256,152,256],[2,256,76,128],[2,256,38,64],[2,256,19,32],[2,256,10,16]]
1. RPN的输入是多层特征图, 通过map会每层特征图分别做cls和reg，做cls,reg本质上就是"调整层数"
   cls调整层数的逻辑：(b,c,h,w) -> (b,x,h,w) 
   reg调整层数的逻辑：(b,c,h,w) -> (b,x,h,w)
"""
rpn = RPN_head(3)        # 每个数据网格点有3个anchors
results = rpn(fpn_outs)  # ([ft_cls0, ft_cls1, ft_cls2, ft_cls3, ft_cls4],
                         #  [ft_reg0, ft_reg1, ft_reg2, ft_reg3, ft_reg4])
    

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
def gen_base_anchors_mine(anchor_base, ratios, scales, ctr=None, scale_major=True):
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
    if ctr is not None:
        x_ctr = ctr[0]
        y_ctr = ctr[1]
    else:
        x_ctr = 0.5 * w
        y_ctr = 0.5 * h
    
    base_anchors = torch.zeros(len(ratios)*len(scales),4)   # (n, 4)
    if scale_major: # 以scale为主，先乘以scale再乘以ratios
        for i in range(len(scales)):
            for j in range(len(ratios)):
                h = (anchor_base * scales[i]).float() * torch.sqrt(ratios[j])
                w = (anchor_base * scales[i]).float() * torch.sqrt(1. / ratios[j])
                index = i*len(ratios) + j
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
        featmap_size(list(float)): (h,wn)
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
"""Q. one stage的SSD在anchor list的生成跟常规two stage有什么不同？
参考ssd如何生成anchor: https://blog.csdn.net/qq_36735489/article/details/83653816
"""
def ssd_get_anchors(anchor_bases, ratios, scales, scale_major=True,
                    featmap_sizes, stride):
    """组合gen_base_anchors()和grid_anchors()来生成all anchors
    ssd的base anchor生成逻辑不太一样，参考：https://www.jianshu.com/p/e13792628bac
    
    
    Args:
        
    Return:
        
    """
    # 1. 生成每张featmaps的base anchors
    base_anchors = []
    for anchor_base in anchor_bases:
        base_anchor = gen_base_anchors_mine(anchor_base, ratios, scales)
        base_anchors.append(base_anchor)
    # 2. 网格化anchors
    all_anchors = []
    for i in range(len(featmap_sizes)):
        all_anchor = grid_anchors_mine(featmap_sizes[i], stride[i], base_anchors[i])
        all_anchors.append(all_anchor)
    # 3. 生成valid flag
    
    

# %%
"""Q.one stage的anchor target的生成跟常规two stage有什么不同?
"""
def ssd_anchor_target():
    pass

# %%
"""Q. 对anchor进行身份指定之前需要对gt bboxes和all anchors进行IOU计算作为评估依据，那如何进行IOU计算？
"""
import torch
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

import numpy as np
def bbox_overlap_new(bb1,bb2):
    """另一个在numpy下的iou计算，逻辑简化了一下更清晰：
    采用一个循环控制其中一个bbox，再借用广播机制和按元素操作来计算另一个bbox组的所有最小值/最大值以及ious
    Args:
        bb1(ndarray), (m,4) [xmin,ymin,xmax,ymax]
        bb2(ndarray), (n,4) [xmin,ymin,xmax,ymax]
    Returns:
        ious(ndarray),(m,n)
    """
    ious = np.zeros((bb1.shape[0], bb2.shape[0]))  # (m,n) 
    
    for i in range(len(bb1)):
        xmax = np.minimum(bb1[i,2],bb2[:,2])
        ymax = np.minimum(bb1[i,3],bb2[:,3])
        xmin = np.maximum(bb1[i,0],bb2[:,0])
        ymin = np.maximum(bb1[i,1],bb2[:,1])
        overlap = ((xmax - xmin)*(ymax - ymin))   # (n,)
        area1 = (bb1[i,2] - bb1[i,0])*(bb1[i,3] - bb1[i,1])  # float
        area2 = (bb2[:,2] - bb2[:,0])*(bb2[:,3] - bb2[:,1])  # (n,)
        
        ious[i] = overlap / (area1 + area2 - overlap)  # 广播机制 (n,)/(float + (n,) - (n,)) = (n,)/(n,) = (n,) 
    return ious

bb1 = np.array([[-20.,-20.,20.,20.],[-30.,-30.,30.,30.]])
bb2 = np.array([[-25.,-25.,25.,25.],[-15.,-15.,15.,15.],[-25,-25,50,50]])
ious_new = bbox_overlap_new(bb1,bb2)


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
    max_overlap, argmax_overlap = overlaps.max(dim=0)      # (n,)对应n个anchors，max_overlap表示每个anchor跟哪一个gt的iou最大 (该变量跟assigned同尺寸，用来给assigned做筛选)
                                                           # argmax_overlap取值(0,1,2)分别代表第0/第1/第2个gt bbox
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
"""Q. 从assigner/sampler得到正样本的anchors后，为什么要做bbox回归？
目的：得到的proposal跟实际gt有偏差，所以希望神经网络能够学到一组参数，帮助proposal更接近gt.
    >为了proposal能够变换到尽可能靠近gt，需要通过两次转换，一次是proposal平移，一次是proposal缩放
    >proposal平移过程公式如下，之所以采用x/y/w/h而不是xmin/ymin/xmax/ymax，是因为可以把变化量减少到两个(x/y)，便于写平移公式
        x' = x + dx*w
        y' = y + dy*h
    >proposal缩放过程公式如下：之所以要加exp是因为一方面非负化，另一方面比较小的dw就能有比较大的对w的缩放效果，利于快速收敛
        w' = w*exp(dw)
        h' = h*exp(dh)
只要定义一个关于(dx,dy,dw,dh)的合适的损失函数，神经网络就能通过学习得到每个proposal所对应的dex/dy/dw/dh，从而变换到更接近gt的形状
"""
def bbox2delta(proposals, gt, means=[0,0,0,0], stds =[1,1,1,1]):
    """对proposal bbox进行回归
    
    Args:
        proposals(tensor): (m,4) 代表(xmin,ymin,xmax,ymax)
        gt(tensor): (m,4) 代表(xmin,ymin,xmax,ymax)
    Return:
        deltas(tensor): (m,4) 代表(dx,dy,dw,dh)
    """
    proposals = proposals.float()
    gt = gt.float()
    # proposal xyxy to xywh
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0
    # gt xyxy to xywh
    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0
    # get dxdydwdh
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)
    # normalize
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas

def delta2bbox(rois,
               deltas,
               means=[0, 0, 0, 0],
               stds=[1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    """在得到deltas的回归结果后，反过来转换成实际bbox坐标
    Args:
        rois(tensor): (m,4)代表proposals，在faster rcnn中m等于2000即会产生2000个proposal
        deltas(tensor): (m, 4*n_classes)代表模型预测的deltas，每个类预测会有4个坐标预测值，所以总列数是4*n_classes(coco为81类，voc为21类，包括背景算其中1类)
    Returns:
        bboxes(tensor): (m, 4*n_classes)
    """
    # denormalize
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    # 
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes

if __name__ == '__main__':
    test_bb2dt = True
    if test_bb2dt:
        delta2bbox()

# %%
"""Q. 从assigner/sampler得到正样本的anchors后，最终如何生成anchor targets?
1. 先得到all_anchors和all_valids

"""
def valid_flags(featmap_size, valid_size, num_base_anchors, device='cuda'):
    """创建合法标签，用来确定哪些位置点是合法的
    由于输入图片pad以后的尺寸作为初始尺寸h/w，变换到特征图尺寸fh,fw过程中，pytorch默认用下取整，也可设置ceil_mode=True选择上取整
    带来的问题是比如pad后图片为300, 8倍下采样，则上取整ceil(300/8)=38, 就说明feat至少要38，放大后才能涵盖原图。
    >如果实际feat_h=37，那合法的feat就是37(可能会有部分原图丢失) 
    >如果实际feat_h=39,则合法feat就是38(多余的feat是没有用的)
    此时定义一个valid_flag，就是把合法feat尺寸上每个点标注成1,额外非法feat上的点标注为0
    (在ssd中没有pad,所以valid size = img size，且设置了ceil_mode=True，所以所有点都合法
    而在faster rcnn中，图片处理事先设置了size divisor，确保能够整除，所以也能让所有点合法)
    
    Args:
        featmap_size(list): 代表一组特征图的尺寸列表比如[(38,38), (19,19), (10,10), (5,5), (3,3), (1,1)]
        valid_size(list): 代表合法尺寸，从(ceil(pad_img_h/stride), feat_h)中间取小值，代表跟原图相关的特征点，而不是超出图片边界的特征点。
        num_base_anchors(int): 代表该层featmap的每一个cell放置多少个base_anchors (比如ssd是4-6个，fasterrcnn是3个)
    Return:
        valid(tensor): (k,) 其中k代表该特征层每一个anchor的合法标志，k=feat_h*feat_w*num_base_anchors
    """
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
    valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
    valid_x[:valid_w] = 1
    valid_y[:valid_h] = 1
    valid_xx = valid_x[None,:].repeat(feat_h, 1).flatten()
    valid_yy = valid_y[:,None].repeat(1, feat_w).flatten()

    valid = valid_xx & valid_yy
    valid = valid[:, None].expand(
        valid.size(0), num_base_anchors).contiguous().view(-1)
    # 跟用repeat效果一样
    # valid = valid[:, None].repeat(1, num_base_anchors).contiguous().view(-1)
    return valid


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
        labels_weights[neg_inds] = 1.0  # labels=1只标记了正样本，label_weights=1则同时指定了正负样本，是为了

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
    """借用inside_flags把得到的data映射回原来的total数据中：(因为原来total数据是堆叠在一起的)
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
得到的anchor targets后，损失分成两部分计算：
1. 第一部分是cls损失，基于rpn_cls预测结果(b,3,h,w)，label，weight进行计算
    >rpn_cls结果先展平，从(b,3,h,w)->(b*3*h*w,1)，同时label为前景=1的标签，
     把问题看成二分类问题(区分前景/背景)，
    >二分类问题采用二值交叉熵损失函数，并送入权重，避免无关样本的loss计入总的loss
     二值交叉熵计算核心：pred(经sigmoid处理过的概率值)和label(二值0，1)，然后做交叉熵运算ylog(y^)+(1-y)log(1-y^)
     每一个label得到一个loss，最后平均loss输出
2. 第二部分是reg损失，基于rpn_reg预测结果(b,12,h,w)，bbox target，weight进行计算
    >rpn_reg结果先展平，从(b,3*4,h,w)->(b*12*h*w,1)，把问题看成4变量回归问题(回归dx,dy,dw,dh)
     把预测的dx,dy,dw,dh与反推出来的dx,dy,dw,dh进行loss计算(反推的dx,dy,dw,dh代表每轮batch的gt bbox与proposal的映射函数)
"""
def weighted_binary_cross_entropy(pred, label, weight, avg_factor=None):
    """带权重二值交叉熵损失函数，用于二分类损失计算：基于head_cls转换输出的(b, out_channle*3, h,w)
    然后进行
    Args:
        pred(tensor): (m,1)代表把特征图经rpn_cls转换后得到的(2,out_channel*3,h,w)的数据拉平到(b*c*h*w,1)
        label(tensor): (m,1)代表是前景则label=1，是背景则label=0，是有代码自己生成的标签
        weight(tensor): (m,1)代表是样本(正样本+负样本)则权值=1, 非样本则权值=0
        avg_factor(int): 代表loss最后平均化缩减的除数，取的是所有样本的个数(多张图就要乘以张数)，比如rpn就是avg_factor=256×2=512
    Return:
        loss(): 代表一个平均损失值
    """
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    return F.binary_cross_entropy_with_logits(
        pred, label.float(), weight.float(),
        reduction='sum')[None] / avg_factor 
# 实例
pred = torch.randn(196992,1)
label = torch.randint(0,1,size=(196992,1))
weight = torch.randint(0,1,size=(196992,1))
avg_factor = 256*2  # 2张图
loss = weighted_binary_cross_entropy(pred,label,weight,avg_factor)


def smooth_l1_loss(pred, target, beta=1.0, reduction='elementwise_mean'):
    """平滑l1损失函数：
    Args:
        pred(tensor): (m,4)代表把特征图经rpn_reg转换后得到的(2,n_anchor*4,h,w)的数据拉平到(b*h*w*c,1)
        target(tensor): (m,4)代表bbox_target, 即预测bbox与gt bbox回归预测的回归参数(dx,dy,dw,dh)
        avg_factor(int): 代表loss最后平均化缩减的除数，取的是所有样本的个数(多张图就要乘以张数)，比如rpn就是avg_factor=256×2=512
    Return:
        loss(): 代表一个平均损失值
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction == 0:
        return loss
    elif reduction == 1:
        return loss.sum() / pred.numel()
    elif reduction == 2:
        return loss.sum()
# 实例
bbox_targets = torch.randint(196992,4)

def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    """带权重平滑l1损失函数：
    Args:
        pred(tensor): (m,4)代表把特征图经rpn_reg转换后得到的(2,n_anchor*4,h,w)的数据拉平到(b*h*w*c,1)
        target(tensor): (m,4)代表bbox_target, 即预测bbox与gt bbox回归预测的回归参数(dx,dy,dw,dh)
        weight(tensor): (m,4)代表bbox的权重，正样本权重=1，其他权重=0
        avg_factor(int): 代表loss最后平均化缩减的除数，取的是所有样本的个数(多张图就要乘以张数)，比如rpn就是avg_factor=256×2=512
    Return:
        loss(tensor): 代表一个平均损失值
    """
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor

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
        num_total_samples(int): 代表正样本+负样本的总和(多张图则要乘以张数)
    Return:
        loss_cls(tensor): (1,)  分类损失
        loss_reg(tensor): (1,)  回归损失
    """
    # 分类损失计算    
    labels = labels.reshape(-1,1)
    labels_weights = labels_weights.reshape(-1,1)
    cls_score = cls_score.permute(0,2,3,1).reshape(-1,1)  # (b,c,h,w) -> (b,h,w,c) -> (b*h*w*c,)
                                                          # 例如对某层特征图(2,3,216,152)->(2,216,152,3)->(x,)
    cls_criterion = weighted_binary_cross_entropy
    loss_cls = cls_criterion(cls_score, labels, labels_weights, 
                             avg_factor=num_total_samples)
    # 回归损失计算
    bbox_targets = bbox_targets.reshape(-1,4)
    
    loss_reg = weighted_smoothl1(bbox_pred, bbox_targets, bbox_weights,
                                 beta = 1./9., avg_factor=num_total_samples)
    return loss_cls, loss_reg


# %%
"""Q.如何从训练网络中得到一定数量的bboxes作为proposals
step1: get_bboxes_single整个过程针对一张图(对应5张特征图)，用于过滤出一定数量的bboxes
注意：此时的过滤输出是相对原图的bbox(xmin/ymin/xmax/ymax)，并且
    >每张特征图先进行nms_pre提取score，输出缩减到2000个(1张特征图)
    >每张特征图进行nms非极大值抑制输出，输出从2000缩减到约1000以内
    >每张特征图再进行nms_post提取(该参数一般取跟nms_pre相同，似乎没什么用)
    >组合5张特征图的nms输出，此时输出会超过2000，比如(3253,5)
    >再进行max_num提取，输出再次缩减到2000个(5张特征图)
    此时就能输出proposals了
    
step2: 对proposals进行第二轮assigner指定和sampler采样，输出512个正负样本

step3: 
    
"""
def get_bboxes_single():
    pass





# %%
"""非极大值抑制的功能？
"""    
import numpy as np
def nms(proposals, iou_thr, device_id=None):
    """实施非极大值抑制: 对输入的一组bboxes进行iou评价，在保留最大置信度基础上去除重叠比较多的框
    这是一个简版的在cpu端运行的nms，由于不断计算iou，速度较慢。
    Args:
        proposal(array): (m,5)代表bbox坐标和置信度，(xmin,ymin,xmax,ymax,score)
        iou_thr(float): 代表iou重叠的阀值，超过该阀值，就认为两个bbox是重叠多余，保留其中置信度高的
    Returns:
        keep(list): 代表proposal中被保留bbox的index
    代码来自rbg神人的github: https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    介绍参考：gloomfish的《对象检测网络中的NMS算法详解》
    1.整个程序过程如下：(基于2大参数，一个不可调参数score阈值，一个可调参数ious阈值)先找到最大置信度的bbox，跟剩余bbox做iou计算，把iou大于阀值的bbox认为是重复比较大的，丢掉，并保留该最大置信度的bbox
    然后从剩余bbox中再找到最大置信度的bbox，再跟剩余bbox做iou计算，把iou大于阀值的bbox认为是重复比较打的，丢掉，并保留该最大置信度的bbox
    通常这个认为是重叠框的iou阀值取0.7
    *所以，本质上就是不断寻找最大置信度的bbox，并丢弃index中跟该bbox的iou非常高的重叠bbox直到index长度=0
     (或者认为是把iou高于阀值iou的这些bbox的置信度都设为0,也就设为没被检测出来)
    *可调参数ious阈值的影响：如果过大会导致去除的少，从而导致大量FP(false positive)，从而导致检测精度下降(因为FP会超过TP，从而正负样本不平衡)
    而如果过小会导致去除的多，从而导致recall大幅下降
    2. 两种nms算法：一种是贪心算法Greedy，一种是最优解算法
    3. 无论one stage还是two stage算法都需要进行nms，只不过two stage是针对proposal进行nms得到rois
    而one stage是针对test来做的，在train的阶段所有anchor都被拿来训练，通过采样原则来获得target

    """
    x1 = proposals[:,0]
    y1 = proposals[:,1]
    x2 = proposals[:,2]
    y2 = proposals[:,3]
    areas = (y2-y1+1) * (x2-x1+1)
    scores = proposals[:,4]
    keep = []
    index = scores.argsort()[::-1]  # 先从大到小排序，返回index
    while index.size >0:
        i = index[0]        
        keep.append(i)     # 每轮循环提取第一个作为对象，并保存   
        x11 = np.maximum(x1[i], x1[index[1:]])    # calculate the points of overlap 
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])        
        w = np.maximum(0, x22-x11+1)    # the weights of overlap
        h = np.maximum(0, y22-y11+1)    # the height of overlap       
        overlaps = w*h        
        ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)        
        idx = np.where(ious<=iou_thr)[0]   # 查找所有ious小于阀值的index保留下来，其他大于阀值的index就相当于丢掉了        
        index = index[idx+1]   # because index start from 1       
    return keep

if __name__=='__main__':
    test_nms = False
    if test_nms:
        # 验证：    
        bboxes=np.array([[100,100,210,210,0.72],
                        [250,250,420,420,0.8],
                        [220,220,320,330,0.92],
                        [100,100,210,210,0.72],
                        [230,240,325,330,0.81],
                        [220,230,315,340,0.9]]) 
        keep = nms(bboxes, 0.7)
        # 绘图：
        import matplotlib.pyplot as plt
        def plot_bbox(dets, c='k', title=None):
            x1 = dets[:,0]
            y1 = dets[:,1]
            x2 = dets[:,2]
            y2 = dets[:,3]    
            plt.plot([x1,x2], [y1,y1], c)
            plt.plot([x1,x1], [y1,y2], c)
            plt.plot([x1,x2], [y2,y2], c)
            plt.plot([x2,x2], [y1,y2], c)
            plt.title(title)
        plt.subplot(121)
        plot_bbox(bboxes,'gray','before nms')   # before nms
        plt.subplot(122)
        plot_bbox(bboxes,'gray')   # before nms
        plot_bbox(bboxes[keep], 'red','after nms')# after nms


# %%
"""改进nms为soft_nms的方法？
1. 常规nms的缺陷，以及改进方法：
    >常规nms对于有相互遮挡的物体也会判定为重叠从而被去除，导致对重叠物体的检测失败
    >通过soft_nms算法可以改进对重叠物体的检测：常规nms是基于si置信度得分进行函数评估，
     并且把所有iou小于阀值iou的bbox置信度都定义成0，而soft_nms则是对该函数进行smooth平滑化，
     也就是加权函数。一种线性加权，一种高斯加权。
     线性加权：对于大于阀值iou的bbox不是直接设置置信度为0，而是更新置信度si=si*(1-iou)
     高斯加权：对于大于阀值iou的bbox不是直接设置置信度为0，而是更新置信度si=si*(exp(-iou^2/sigma))
     加权以后的好处是：对于重叠度非常高的，置信度得分会变得很低，而对于部分重叠的两物体，则置信度不会被调的很低
     从而保证重叠度非常高的框会被去除，而部分重叠的则保留。
     同时voc数据集遮挡不多所以softnms影响不大, 但coco遮挡较多所以影响会相对大一些。
    >当前soft_nms能够带来平均1%的AP提升，但soft_nms依然受自定义iou_thr影响较大，需要手工调整iou_thr的值，还有改进的空间
     如果能够变成可学习的就好了
"""
import numpy as np
# 对soft_nms的验证
def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    """Soft NMS implementation.
    代码来源：https://blog.csdn.net/jacke121/article/details/82795272
    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx
    Args:
        box_scores: (N, 5) boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: 高斯加权置信度重计算参数，scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: 保留前k个结果，如果k<=0则保留所有结果.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        max_score_index = torch.argmax(box_scores[:, 4])   # 获得置信度排序的index
        cur_box_prob = torch.tensor(box_scores[max_score_index, :])
        picked_box_scores.append(cur_box_prob)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])  # 该句需要改为iou检测
        
        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.tensor([])

if __name__=='__main__':
    test_soft_nms = False
    if test_soft_nms:
        # 验证：    
        bboxes=np.array([[100,100,210,210,0.72],
                        [250,250,420,420,0.8],
                        [220,220,320,330,0.92],
                        [100,100,210,210,0.72],
                        [230,240,325,330,0.81],
                        [220,230,315,340,0.9]]) 
        keep = soft_nms(bboxes, 0.7)
        # 绘图：
        import matplotlib.pyplot as plt
        def plot_bbox(dets, c='k', title=None):
            x1 = dets[:,0]
            y1 = dets[:,1]
            x2 = dets[:,2]
            y2 = dets[:,3]    
            plt.plot([x1,x2], [y1,y1], c)
            plt.plot([x1,x1], [y1,y2], c)
            plt.plot([x1,x2], [y2,y2], c)
            plt.plot([x2,x2], [y1,y2], c)
            plt.title(title)
        plt.subplot(121)
        plot_bbox(bboxes,'gray','before nms')   # before nms
        plt.subplot(122)
        plot_bbox(bboxes,'gray')   # before nms
        plot_bbox(bboxes[keep], 'red','after nms')# after nms



# %%
"""如何得到rois?
把前面获得bbox proposal放在一起考虑，就是如何从特征图里获得对应bbox作为rois
rois的得到过程：
    >通过all_anchors从cls_score/bbox_pred中提取置信度高的bboxes作为proposal(3000+, 5)
    >通过nms非极大值抑制去除重叠率高的bbox后得到更新的proposal(2000,5)
    >通过指定和采样得到更新的proposal(512,4)
    >通过bbox2roi把得到proposal转换成roi的格式(增加第一列置信度=0)
"""
# 提取置信度高的bboxes，就是get_bboxes_single()
# 非极大值抑制就是nms()
# 制定和采样就是assigner()和sampler()
# bbox2roi()
# 联合调试：


# %%
"""Q. 如何基于rois提取bbox_feats? 也就是如何设计RoIAlign？
1.设计RoIAlign的目的：从backbone出来的feats对应到不同bbox中有不同的尺寸，通过RoIAlign可以把每个bbox包含的feats
尺寸都调整成一个尺寸(7,7)，层数也不改变，这样做的原因是接下来做分类回归的网络只能接收size相同features，而不能是size变化的features
所以需要RoiAlign把特征尺寸都调为一样的。

2.需要利用roi extractor实现，里边核心的部分就是RoIAlign模块：
    >数据准备：把rois先map到每一层特征图上：也就是rois(512,4) -> [(4,5),(12,5),(77,5),(931,5)]
    >数据对应：把4组rois和4组feats送入RoIAlign，相当于基于每个rois从特征图中抠图，并统一成7x7尺寸，层数不变
     所以每一个roi对应出一个抠图(256,7,7), 512个rois对应抠图就是(512，256,7,7)，如果是2张图就是(1024,256,7,7)这就是roi_feats

3. 底层原理：先把每个rois映射到对应特征图上，然后把对应特征划分为7x7的49块，然后在每一块中获得一个最大值作为池化输出值，最终得到7x7的尺寸
底层RoiAlign的实现源码：rois映射到特征图上时除以stride时保留浮点数，划分7x7小块大小时也保留浮点数，对每一块获取一个最大值采用双线性采样来获得
对比RoiPooling的实现原理：rois映射到特征图上时除以stride时下取整，划分7x7小块大小时下取整，对每一块获取一个最大值采用直接     
一个实例：

"""
def roi_extractor():
    roi_feats = []
    return roi_feats


# %%
"""Q. RoIPooling与RoIAlign的区别的联系？
1. 两者的功能：都是基于rois和feats
2. RoIPooling
    >过程：
    >缺陷：在区域划分时采取的取整测策略会丢失部分像素，所然能直接做max pool操作，但对小物体识别的精度有影响
3. RoIAlign
    >过程：
    >优点：在区域划分时采取的保留小数，然后配合双线性插值法获得每个点位的像素，再做max pool操作

"""


# %%
"""Q.如何定义bbox head对roi_feats进行
由于roi_feats(也称为bbox_feats的维度是(1024,256,7,7))，其中1024就是rois的个数，每个rois对应了从特征图抠取的一块特征数据
"""




# %%
"""one stage的SSD head有什么特点？
1. 分类回归全部用卷积来做，没有全连接，所以参数数量和计算量都少了很多
2.
"""
import torch.nn as nn
class SSDHead(nn.Module):
    """做一个完整的SSD Head，调用前面已经实现的SSD get_anchors()/anchor target()
    """
    def __init__(self,  in_channels, out_channels):

        
out_channels = [()]



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
整个流程跟forward很像，但forward得到cls_score/bbox_pred后就进行loss计算了，
而test计算是在得到cls_score/bbox_pred之后进行进一步转换：
1. 分类：cls_score(2000,81)，先对每行进行求和归一化然后送入softmax转换成概率scores(2000,81)
2. 回归：bbox_pred(2000,4*81), 把bbox_pred作为delta转换为x/y
"""
def get_test_result():
    """测试模拟在获得cls_score/bbox_pred之后的处理过程
    """
    # 分类处理
    
    # 回归处理
    
    # nms
    
    # result获得
    result = []
    return result



# %%
"""Q. 如何对model效果进行test和评估？
注意1：测试时cfg/model/weight三者需要匹配，由于大部分mmdetection所提供的weights都是基于coco进行训练的，
所以测试所搭建的模型需要采用针对coco数据集的模型和cfg文件(只有少部分SSD/FasterRcnn提供了voc版本的weight)
下载weights的网址：https://github.com/open-mmlab/mmdetection/blob/master/MODEL_ZOO.md
"""
import torch
import cv2
import mmcv
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.datasets.transforms import ImageTransform
from mmdet.core import get_classes
import numpy as np
from B03_dataset_transform import vis_bbox

def test_img(img_path, config_file, class_name='coco', device = 'cuda:0'):
    """测试单张图片：相当于恢复模型和参数后进行单次前向计算得到结果
    注意由于没有dataloader，所以送入model的数据需要手动合成img_meta
    1. 模型输入data的结构：需要手动配出来
    2. 模型输出result的结构：
    Args:
        img(array): (h,w,c)-bgr
        config_file(str): config文件路径
        device(str): 'cpu'/'cuda:0'/'cuda:1'
        class_name(str): 'voc' or 'coco'
    """
    # 1. 配置文件
    cfg = mmcv.Config.fromfile(config_file)
    cfg.model.pretrained = None
    # 2. 模型
    path = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
    model = build_detector(cfg.model, test_cfg = cfg.test_cfg)
    _ = load_checkpoint(model, path)
    model = model.to(device)
    model.eval()             # 千万别忘了这句，否则虽能出结果但少了很多
    # 3. 图形/数据变换
    img = cv2.imread(img_path)
    img_transform = ImageTransform(size_divisor = cfg.data.test.size_divisor,
                                   **cfg.img_norm_cfg)
    # 5. 数据包准备
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(img, scale= cfg.data.test.img_scale)
    img = torch.tensor(img).to(device).unsqueeze(0) # 应该从(3,800,1216) to tensor(1,3,800,1216)
    
    img_meta = [dict(ori_shape=ori_shape,
                     img_shape=img_shape,
                     pad_shape=pad_shape,
                     scale_factor = scale_factor,
                     flip=False)]

    data = dict(img=[img], img_meta=[img_meta])
    # 6. 结果计算: result(a list with )
    # 进入detector模型的forward_test(),从中再调用simple_test(),从中再调用RPNTestMixin类的simple_test_rpn()方法
    # 测试过程：先前向计算得到rpn head的输出，然后在rpn head中基于该输出计算proposal_list(一张图(2000,5))
    
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    # 7. 结果显示
    class_names = get_classes(class_name)
    labels = [np.full(bbox.shape[0], i, dtype=np.int32) 
        for i, bbox in enumerate(result)]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    scores = bboxes[:,-1]
    img = cv2.imread(img_path, 1)
    

    vis_bbox(img.copy(), bboxes, label=labels, score=scores, score_thr=0.7, 
             label_names=class_names,
             instance_colors=None, alpha=1., linewidth=1.5, ax=None)
    

if __name__ == "__main__":     
    test_this_img = True
    if test_this_img:
        img_path = 'test/test_data/test13.jpg'    
        config_file = 'test/test_data/cfg_fasterrcnn_r50_fpn_coco.py'
        class_name = 'coco'
        test_img(img_path, config_file, class_name=class_name)


# %%
"""Q. 如何对整个数据集进行测试评估？

"""
import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors


def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')  # 字符串，结果文件路径
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')  # bool变量，指明是否调用模型的show_result()显示结果
    args = parser.parse_args()
    return args


def main():
    """针对faster rcnn在voc的评估做微调
    1. args parse用直接输入替代
    2. 
    """
#    args = parse_args()

#    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
#        raise ValueError('The output file must be a pkl file.')
    config_path = './cfg_fasterrcnn_r50_fpn_coco.py'
    checkpoint_path = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
    cfg = mmcv.Config.fromfile(config_path)
    out_file = 'dataset_eval_result/results'  # 是否输出结果文件, 后边会添加.json后缀
    eval_type = 'proposal_fast'
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    if cfg.gpus == 1:
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, checkpoint_path)
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader, show=False)
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(detectors, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            checkpoint_path,
            dataset,
            _data_func,
            range(cfg.gpus),
            workers_per_gpu=cfg.proc_per_gpu)
    # debug
    out_file = 'test1.json'
    outputs = dict(a=1,b=2,c=3)
    if out_file:
        print('writing results to {}'.format(out_file))  
        mmcv.dump(outputs, out_file)  # 先把模型的测试结果输出到文件中: 如果文件不存在会创建
        
        eval_types = eval_type
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = out_file
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_file = out_file + '.json'
                    results2json(dataset, outputs, result_file)
                    coco_eval(result_file, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = out_file + '.{}.json'.format(name)
                        results2json(dataset, outputs_, result_file)
                        coco_eval(result_file, eval_types, dataset.coco)

if __name__ == '__main__':
    main()

# %%
"""Q. 如何计算训练的均值平均精度mAP和召回率recall
参考：https://blog.csdn.net/hysteric314/article/details/54093734 (浅显的计算实例recall/precision)
参考：https://blog.csdn.net/l7H9JA4/article/details/80745028 (完整描述AP/mAP的由来)
1. TP/FP/FN的概念：
    >TP(true positive)真阳性，是指模型检测为正，gt确实为正
    >FP(false postive)假阳性，是指模型检测为正，gt但是为负
    >TN(true negative)真阴性，是指模型检测为负，gt确实为负
    >FN(false negative)假阴性，是指模型检测为负，gt但是为正

2. 准确率Accuracy: 
    Accuracy = (TP + TN) / (TP+FP+TN+FN)
    
3. 精确率Precision: (理解为一把抓到多少只真猫)正样本预测正确率在预测出来的正样本中比率，侧重在自己追求成功率(所以也称查准率)，
   相当于"10只预测"找到3只真猫, 0.3的精度只用了10次预测 
    Precision = TP / (TP + FP)
   又包含两个子概念：来源于pascal voc (http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf 第4.2章节)
       >AP: 平均精度，可以总结PR曲线，是指
       >mAP: 均值平均精度

4. 召回率Recall: (理解为真猫召回recall多少只)正样本预测正确率在所有正样本中比率，侧重在追求完成整个任务(所以也称查全率)，
   相当于"10只真猫"找到3只真猫， 0.3的recall有可能用了200次预测
    Recall = TP / (TP + FN)

注意1： 准确率指标如果对于detection这类样本天然不平衡的应用，准确率就失效了，只能用精确率。
比如背景样本本来就很多，分类器只需要预测所有样本都是背景，准确率就上去了，此时只有用精确率来评估是合理的。
而recall/precision都能够针对单一正样本，所以都适合用在不平衡样本的任务上，也就有了P-R曲线。

注意2：以上Precise/Recall是两个相互影响的指标，一个高另一个就会变低。
所以就有PR曲线来显示两者此消彼长的关系。
在疾病检测要的是准确性，所以看中precision，而做搜索要得是全面获得，更看中召回率。

5. 由于precision和recall都有单点值局限性，所以提出mAP来反映全局性能的指标。
均值平均精度mAP：

常规计算过程
1. 先计算iou，然后假定iou>iou_thr(比如0.5)为正样本TP(这里0.5是pascal voc的指标，而coco是建议对不同iou分别进行计算)
从而也能得到FP(错误检测认为是正样本), 基于TP,FP,FN计算precision和recall
2. pascal voc定义从[0,0.1,0.2,...1]共计11个置信度阈值，在每个置信度阈值之上计算precision, 然后取11个平均值就是mAP
但上面是2007 voc的mAP方法，在2010后改为采用所有数据点而不是11个置信度阈值点计算AP, 然后所有类别平均值就是mAP
3. 

"""



# %%
"""除了mAP/AP/AR之外，还有一个运行速度参数fps如何理解？
1. fps是指frame per second帧/秒，用来表示模型每秒能够处理的图片数。
也可以叫做频率，用hz来做单位。在物体检测方向上fps跟hz的作用是一摸一样的。
2. 在图像领域fps有一个阀值，如果要做视频方向的检测，图像处理速度需要在25-30fps
低于25fps就会产生人眼所见的频闪了，而高清视频的fps也在60fps以下。因此对于检测器
的速度只有超过30fps才适合用来做视频检测：基本上当前two stage的detector的fps都在10以下
而one stage中有部分检测器fps超过30了，比如ssd300
"""






# %%
"""Q. 如何实施视频物体检测？
参考：https://github.com/amdegroot/ssd.pytorch/blob/master/demo/live.py
"""
from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                            FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    stream = WebcamVideoStream(src=0).start()  # default camera
    time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    while True:
        # grab next frame
        frame = stream.read()
        key = cv2.waitKey(1) & 0xFF

        # update FPS counter
        fps.update()
        frame = predict(frame)

        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', frame)
        if key == 27:  # exit
            break
if __name__ == "__main__":     
    test_video_detect = True
    if test_video_detect:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        
        from data import BaseTransform, VOC_CLASSES as labelmap
        from ssd import build_ssd
        
        net = build_ssd('test', 300, 21)    # initialize SSD
        net.load_state_dict(torch.load(args.weights))
        transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
        
        fps = FPS().start()
        cv2_demo(net.eval(), transform)
        # stop the timer and display FPS information
        fps.stop()
        
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        
        # cleanup
        cv2.destroyAllWindows()
        stream.stop()


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


