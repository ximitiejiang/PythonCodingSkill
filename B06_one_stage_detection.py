#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:18:35 2019

@author: ubuntu
"""

# %% 基于RetinaNet重新推演一下one stage detector的过程
"""
one stage detector的演进：
可参考: https://github.com/open-mmlab/mmdetection/blob/pytorch-0.4.1/MODEL_ZOO.md
        https://github.com/hoya012/deep_learning_object_detection 
可参考: https://arxiv.org/pdf/1811.04533.pdf 这篇2019年M2Det的论文详细对比了mAP和fps，
       貌似速度上SSD依然最快，其次RefineDet/M2Det；而精度上M2Det最高，其次CornerNet
2. One stage:
    > (2014)最早有multibox的概念
    > (2016)产生SSD和Yolo：             SSD(fps=29.2@1GPU, mAP=25.7@Vgg16)
    > (2017)产生RetinaNet和yolo v2:     RetinaNet(fps=9.1@8GPU, mAP=35.6@Resnet50)
    > (2018)产生CornerNet:              CornerNet(mAP=42.1)
    > (2019)产生M2Det:                  M2Det(mAP=44.2)
"""


# %% 
"""Q. SSD的anchor target是如何计算的？跟two stage的faster rcnn有什么区别？
"""
def anchor_target_ssd():
    """one stage ssd的anchor target所做的事情跟tow stage类似
    输入相同：都是anchor_list(m,4), gt_bbox(n,4), valid_list(m,4)
    1. assigner: 对每一个anchor指定身份，基于iou的算法原则，生成assigned_list(m,)，依然是m个anchors
    2. sampler: two stage是采用随机采样获得总和256个正负样本，很大程度上克服了样本不平衡的问题
        而在ssd这块采用pseudo sampler也就是假采样，只是从8000多个anchor中提取的正样本和负样本，
        但由于总和依然是8000多个，没有对样本不平衡问题有改善。
    3. anchor_target: 获得label和label_weight，这里的区别在于
        >two stage的anchor head不需要label, 因为只是做前景背景的一个判断，相当与二分类，
        所以只是把正负样本的label=1,无关样本=0。而在ssd这块anchor head实际是作为bbox head使用，
        所以是需要准确label用于后边的分类损失计算，所以label取的是正确的原始值。
        label_weight这块一样都是把正负样本的weight=1，同时bbox也转化为delta用于后边的回归损失计算，没有区别
        >two stage的anchors是把多张特征图汇总到一起计算的，所以有一步unmap
        而ssd不需要unmap操作
    """
    # part1: assigner: 对所有anchor指定身份
    
    # part2: pseudo sampler: 获得正负样本index
    
    # anchor target generate: 生成weight，转换bbox2delta
    
    
    
# %%
"""Q. SSD的loss计算是如何操作的？跟two stage的loss计算有什么区别？
"""
def loss():
    """SSD在计算loss时主体跟two stage一样，分成cls_loss和reg_loss两部分
    1. cls_loss：选择的也是多分类的交叉熵进行损失计算，区别在于
        >由于存在样本不平衡问题(送进来的正负样本之和为8732, 负样本太多)，所以计算损失后
         并没有进行累加缩减，而是先对loss进行排序，取出其中loss最大的一部分，使正负样本
         的数量比为1：3(可设置，在cfg里是neg_pos_ratio=3)，然后在求正负loss的平均值
         作为cls_loss
    2. reg_loss：
    
    """
