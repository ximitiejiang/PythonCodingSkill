#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 12:08:05 2019

@author: ubuntu
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
 

def imshow(img, win_name='img', wait_time=0):
    """用cv2.imshow()显示
    Args:
        img(array): (h,w,c)

    """
    assert len(img.shape) == 3 # only support (h,w,c), not support (w, h)
    cv2.imshow(win_name, img)
    cv2.waitKey(wait_time)  # 0 means showing img until key pressed


def imshow_bbox(img, 
                bboxes,
                labels=None,
                class_names=None,
                b_colors=(255,0,0),
                t_colors=(255,0,0),
                font_scale=0.5,
                thickness=2):
    """show img，bboxes, labels
    
    Args:
        img(array): img (h,w,c) or (w,h)
        bboxes(list): [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
        labels(list): [label_1, label_2,...]
    """
#    img = img.astype(np.int32)
    
    if labels is None:
        for bbox in bboxes:
            bbox = bbox.astype(np.int32) #bbox在绘制时需要int
            left_top = (bbox[0], bbox[1])
            right_bottom = (bbox[2], bbox[3])
            cv2.rectangle(
                img, left_top, right_bottom, b_colors, thickness=thickness)
            
    else: # labels is not None:
        for bbox,label in zip(bboxes,labels):
            bbox = bbox.astype(np.int32) #bbox在绘制时需要int
            left_top = (bbox[0], bbox[1])
            right_bottom = (bbox[2], bbox[3])
            cv2.rectangle(
                img, left_top, right_bottom, b_colors, thickness=thickness)
            
            label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
            cv2.putText(img, label_text, (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, t_colors)
    plt.imshow(img)
