#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:24:32 2019

@author: ubuntu
"""
            
# %%
'''Q. backbones有哪些，跟常规network有什么区别？
'''    
# resnet作为backbone的修改


# 普通VGG16： blocks = [2,2,3,3,3]每个block包含conv3x3+bn+relu，算上最后3层linear总共带参层就是16层，也就是vgg16名字由来
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
    
channels = [256,512,1024,2048]
sizes = [(152,256),(76,128),(38,64),(19,32)]
feats = []
for i in range(4):  # 构造假数据
    feats.append(torch.randn(2, channels[i], sizes[i][0], sizes[i][1]))

fpn = FPN_neck()
outs = fpn(feats)  # 输出5组特征图(size为阶梯状分辨率，152x256, 76x128, 38x64, 19x32, 10x16)
                   # 该输出特征size跟输入图片尺寸相关，此处256-128-64-32-16是边界
    
# ssd detection中，ssd head获得的多尺度特征图直接来自VGG16(但VGG16需要做微调)
channels = [512, 1024, 512, 256, 256, 256]
sizes = [(38,38),(19,19),(10,10),(5,5),(3,3),(1,1)]                   

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
要生成每一层特征层需要的anchors，分成如下几步：
1. 定义anchor base size：这步定义的base size=[4,8,16,32,64]取决于特征图的size大小，
   比如256边界的anchor size取4，128边界取8，64边界取16, 
2. 创建base anchors
3. 
"""
import torch
def gen_base_anchors(base_size, ctr, ratios, scale_major, scales):
    """从anchor_generator类中提取出如下函数,对interface做了微调，可以对不同类型的特征层生成不同尺寸的anchor组
    如FPN出来的5层特征，可以生成5组anchors
    Args:
        base_size(list): [4,8,16,32,64]
        ctr()
    Returns:
        base_anchors()
    """
    w = base_size
    h = base_size
    if ctr is None:  #如果不输入中心点，则取w/2，h/2为中心点
        x_ctr = 0.5 * (w - 1)
        y_ctr = 0.5 * (h - 1)
    else:
        x_ctr, y_ctr = ctr
    h_ratios = torch.sqrt(ratios)  #
    w_ratios = 1 / h_ratios
    if scale_major:
        ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
    else:
        ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
        hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)
    base_anchors = torch.stack(
        [
            x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
        ],
        dim=-1).round()
    return base_anchors

anchors = gen_base_anchors()


# %%    anchor的三部曲：(base anchor) -> (anchor list) -> (anchor target)
"""Q.如何产生anchor list?
"""

# %%    anchor的三部曲：(base anchor) -> (anchor list) -> (anchor target)
"""Q.如何筛选出anchor target?
"""


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
"""



# %%
'''Q. 对比SingleStageDetector与TwostageDetector是如何抽象出来的？
1. 共用的部分(base的部分)：init_weights(), forward_test(), forward(), show_result()
2. 独立重写的部分(抽象方法的部分)： init(), extract_feat(), forward_train(), simple_test(), aug_test()
3. single stage detector: 一般通过backbone输出多路，经FPN neck

   two stage detector：一般通过backbone输出多路，经
'''
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from . import builder
# 先看BaseDetector: 
class BaseDetector(nn.Module):
    """base detector作为一个基类来定义的"""
    __metaclass__ = ABCMeta
    def __init__(self):
        super().__init__()
    @abstractmethod
    def extract_feat(self, imgs):
        """这是强制需要定义的方法"""
        pass
    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        pass
    def simple_test(self, img, img_meta, **kwargs):
        pass
    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

# 再看RPN detector
class RPN(BaseDetector, RPNTestMixin):
    """RPN detector
    Process: backbone -> neck -> rpn_head
    """
    def __init__(self,backbone,neck,rpn_head,
                 train_cfg,test_cfg,pretrained=None):
        super(RPN, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck) if neck is not None else None
        self.rpn_head = builder.build_head(rpn_head)
        self.init_weights(pretrained=pretrained)
    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    def forward_train(self, img, img_meta, gt_bboxes=None):
        x = self.extract_feat(img)
        rpn_outs = self.rpn_head(x)
        rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
        losses = self.rpn_head.loss(*rpn_loss_inputs)
        return losses
    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn)
        if rescale:
            for proposals, meta in zip(proposal_list, img_meta):
                proposals[:, :4] /= meta['scale_factor']
        # TODO: remove this restriction
        return proposal_list[0].cpu().numpy()
    def aug_test(self, imgs, img_metas, rescale=False):
        pass

# 再看single stage detector
class SingleStageDetector(BaseDetector):
    """singlestage detector用于给ssd/yolo
    Process: backbone -> neck -> bbox_head
    """
    def __init__(self, backbone, neck=None, bbox_head=None,
                 train_cfg=None, test_cfg=None, pretrained=None):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.init_weights(pretrained=pretrained)
    def extract_fact(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(*loss_inputs)
        return losses
    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list]
        return bbox_results[0]
    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
        
# 再看two stage detector
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin, MaskTestMixin):
    """twostage detector用于给fasterrcnn
    Process: backbone -> rpn_head -> 
    """
    def __init__(self, backbone, neck=None, rpn_head=None, bbox_head=None, mask_roi_extractor=None,
                 mask_head=None, train_cfg=None, test_cfg=None, pretrained=None):
        super().__init__()
        self.backbone=builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.rpn_head = builder.build_head(rpn_head)
        self.bbox_roi_extractor = builder.build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = builder.build_head(bbox_head)
        self.mask_roi_extractor = builder.build_roi_extractor(mask_roi_extractor)
        self.mask_head = builder.build_head(mask_head)
        self.init_weights(pretrained=pretrained)
    def extract_fact(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    def forward_train(self,img,img_meta,gt_bboxes,gt_bboxes_ignore,gt_labels,
                      gt_masks=None,proposals=None):
        x = self.extract_feat(img)
        losses = dict()
        if self.with_rpn:
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
        if self.with_bbox or self.with_mask:  # 准备sampling results
            sampling_results= []
        if self.with_bbox:              # 计算roi,计算target,计算loss
            rois = bbox2roi()
            bbox_feats = self.bbox_roi_extractor()
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            bbox_targets = self.bbox_head.get_target()
            loss_bbox = self.bbox_head.loss()
        if self.with_mask:
            pos_rois = bbox2roi()
            mask_feats = self.mask_roi_extractor()
            mask_pred = self.mask_head()
            mask_targets = self.mask_head.get_target()
            loss_mask = self.mask_head.loss()
        return losses


























