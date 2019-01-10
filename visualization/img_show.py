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



def img_bbox_label_show(img, bboxes=None,labels=None):
    """show img，bboxes, labels(with scores)
    
    Args:
        img(array): img (h,w,c) or (w,h)
        bboxes(list): [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
        labels(list): [label_1, label_2,...]
    """
    if bboxes is not None:
#        plt.imshow(img)
        for bbox in bboxes:
            bbox = bbox.astype(np.int32) #bbox在绘制是需要是int
            left_top = (bbox[0], bbox[1])
            right_bottom = (bbox[2], bbox[3])
            colors=(255,0,0)
            thickness= 2

            cv2.rectangle(
                img, left_top, right_bottom, colors, thickness=thickness)    
    plt.imshow(img)
    # TODO: show labels with score
    if labels:
        pass
