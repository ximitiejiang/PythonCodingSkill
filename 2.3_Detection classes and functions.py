#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:24:32 2019

@author: ubuntu
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
            
# %%
'''Q. backbones有哪些，跟常规network有什么区别？
'''    
# resnet


# %%
'''Q. neck有哪些，特点是什么？
1. FPN先采用1x1，用于
2. 
'''
# FPN neck




# %%
'''Q. head有哪些，特点是什么？
'''
# ssd: 采用bbox head



# %%
'''Q.如何产生base anchors?
参考：
1. 
'''
import torch
def gen_base_anchors(base_size, ctr, ratios, scale_major, scales):
    """从anchor_generator类中提取出如下函数,对interface做了微调
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





























