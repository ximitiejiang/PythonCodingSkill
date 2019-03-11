#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 08:26:28 2019

@author: ubuntu
"""

import torch
import cv2
import mmcv
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.datasets.transforms import ImageTransform
from mmdet.core import get_classes
import numpy as np
from matplotlib import pyplot as plt
from dataset.utils import vis_bbox


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
        img_path = 'test12.jpg'    
        config_file = 'cfg_fasterrcnn_r50_fpn_coco.py'
        class_name = 'coco'
        test_img(img_path, config_file, class_name=class_name)
