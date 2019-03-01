#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:44:50 2019

@author: ubuntu
"""


from B03_dataset_transform import VOCDataset, imshow_bboxes_labels

data_root = 'data/VOCdevkit/'
ann_file=[data_root + 'VOC2007/ImageSets/Main/trainval.txt',
          data_root + 'VOC2012/ImageSets/Main/trainval.txt']
img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/']

voc07 = VOCDataset(ann_file[0], img_prefix[0])
voc12 = VOCDataset(ann_file[0], img_prefix[0])
dataset = voc07 + voc12             # Dataset类有重载运算符__add__，所以能够直接相加 (5011+5011)
classes = voc07.CLASSES
img_data = dataset[29]               # len = 10022

imshow_bboxes_labels(img_data.img, img_data.bboxes, img_data.labels,
                     class_names = classes)