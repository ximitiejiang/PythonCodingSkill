#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:05:16 2019

@author: ubuntu
"""
import logging
from ssdvgg import SSDVGG
from ssd_head import SSDHead
import torch.nn as nn
import numpy as np
import mmcv

class OneStageDetector(nn.Module):
    """one stage单级检测器
    1. 采用ssd head作为bbox head来使用： bbox head的本质应该是输出最终结果。
    虽然ssd head继承自anchor head但他并没有用来生成rois，所以作为bbox head使用。
    2. 
    """
    def __init__(self, cfg, pretrained=None):  # 输入参数修改成cfg，同时预训练模型参数网址可用了
        super(OneStageDetector, self).__init__()
        self.backbone = SSDVGG(**cfg.model.backbone)        # 初始化backbone - 参数要导入
        self.bbox_head = SSDHead(**cfg.model.bbox_head)      # 初始化bbox head
        
        self.train_cfg = cfg.train_cfg
        self.test_cfg = cfg.test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))
            
        self.backbone.init_weights(pretrained=pretrained)
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(*loss_inputs)
        return losses
    
    def forward_test(self, imgs, img_metas, **kwargs):
        """用于测试时的前向计算：如果是单张图则跳转到simple_test(), 
        如果是多张图则跳转到aug_test()，但ssd当前不支持多图测试(aug_test未实施)
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)
    
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)
    
    
    def simple_test(self, img, img_meta, rescale=False):
        """用于测试时单图前向计算：
        输出
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        # TODO: 检查bbox_head的get_bboxes()
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        # TODO: replace bbox2result()
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]
    
    def aug_test(self, imgs, img_metas, rescale=False):
        """用于测试时多图前向计算: 当前ssd不支持多图测试"""
        raise NotImplementedError
    
    def show_result(self, data, result, img_norm_cfg,
                    dataset='coco',
                    score_thr=0.3):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        # TODO: replace tensor2imgs()
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)

        if isinstance(dataset, str):
            # TODO: replace get_classes()
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)) or dataset is None:
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            bboxes = np.vstack(bbox_result)
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    # TODO: delete mask的部分是否必要
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr)