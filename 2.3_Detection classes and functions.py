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


# VGG作为backbone的修改




# %%
"""如何从backbone获得multi scale多尺度不同分辨率的输出
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
2. 方式2：通过
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
    
    
# 


# %%
'''Q. head有哪些，特点是什么？
'''
# ssd: 采用ssd head (继承自bbox head)



# %%
'''Q.如何产生base anchors?
参考：
1. 
'''
import torch
def gen_base_anchors(base_size, ctr, ratios, scale_major, scales):
    """从anchor_generator类中提取出如下函数,对interface做了微调，可以对不同类型的特征层生成不同尺寸的anchor组
    如FPN出来的5层特征，可以生成5组anchors
    Args:
        d
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


# %%
"""Q.如何产生base anchors?
"""

# %% 
"""Q.如何筛选anchors?
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


























