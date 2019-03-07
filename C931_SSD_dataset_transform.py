#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:44:50 2019

@author: ubuntu
"""


from B03_dataset_transform import VOCDataset, CocoDataset, vis_bbox

source = 'coco'

if source == 'voc':
    data_root = 'data/VOCdevkit/'
    ann_file=[data_root + 'VOC2007/ImageSets/Main/trainval.txt',
              data_root + 'VOC2012/ImageSets/Main/trainval.txt']
    img_prefix=[data_root + 'VOC2007/', 
                data_root + 'VOC2012/']
    
    voc07 = VOCDataset(ann_file[0], img_prefix[0])
    voc12 = VOCDataset(ann_file[0], img_prefix[0])
    dataset = voc07 + voc12             # Dataset类有重载运算符__add__，所以能够直接相加 (5011+5011)
    classes = voc07.CLASSES
    img_data = dataset[8679]               # len = 10022
    
    vis_bbox(img_data.img, img_data.bboxes, img_data.labels, label_names = classes)


if source == 'coco':
    data_root = 'data/coco/'
    ann_file=[data_root + 'annotations/instances_train2017.json',
              data_root + 'annotations/instances_val2017.json']
    img_prefix=[data_root + 'train2017/',
                data_root + 'val2017/']
    
    trainset = CocoDataset(ann_file[0], img_prefix[0])
    valset = CocoDataset(ann_file[1], img_prefix[1])
    classes = trainset.CLASSES
    img_data = trainset[0]
    vis_bbox(img_data.img, img_data.bboxes, img_data.labels, label_names = classes)