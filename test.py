##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Mon Apr 22 16:50:54 2019
#
#@author: ubuntu
#"""
#
#

import cv2
import numpy as np
# 创建一张全0的200×200的黑色图片，中间放一个100×100的白色方块
img = np.zeros((200, 200), dtype=np.uint8)
img[50:150, 50:150] = 255

# 二值化
ret, thresh = cv2.threshold(img, 127, 255, 0)
# findcontours只接受黑白图，也就是先要对图片先灰度化，再二值化
image, contours, hierarchy = cv2.findContours(thresh, 
                                              cv2.RETR_TREE, 
                                              cv2.CHAIN_APPROX_SIMPLE)
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color, contours, -1, (0,255,0),2)
cv2.imshow('contours', color)
cv2.waitKey()
cv2.destroyAllWindows()



#import numpy as np
#from matplotlib import pyplot as plt
#
#"""
#一张图片，缩小到(1333,800),相应的gt_bbox也缩小到
#所以对小物体的预测，主要取决于浅层anchor的大小，
#分析过程：
#1. 假设retinanet针对voc数据集，在没有别的变换情况下，输出的bbox的分布主要是中等尺寸(1024-9216)和大尺寸的bbox(>9216), 小尺寸bbox(<1024)非常少
#    此时retinanet的base anchors是足够覆盖voc的中等尺寸和大尺寸anchor
#2. 假设retinanet针对coco数据集
#"""
#
#"""这是retinanet最浅层0层的base_anchors, 最小面积945，最大面积2485"""
#ba0 = np.array([[-19.,  -7.,  26.,  14.],
#                [-25., -10.,  32.,  17.],
#                [-32., -14.,  39.,  21.],
#                [-12., -12.,  19.,  19.],
#                [-16., -16.,  23.,  23.],
#                [-21., -21.,  28.,  28.],
#                [ -7., -19.,  14.,  26.],
#                [-10., -25.,  17.,  32.],
#                [-14., -32.,  21.,  39.]])
#"""retinanet第1层base_anchors, 最小面积3969，最大面积10,201"""
#ba1 = np.array([[-37., -15.,  52.,  30.],
#                [-49., -21.,  64.,  36.],
#                [-64., -28.,  79.,  43.],
#                [-24., -24.,  39.,  39.],
#                [-32., -32.,  47.,  47.],
#                [-43., -43.,  58.,  58.],
#                [-15., -37.,  30.,  52.],
#                [-21., -49.,  36.,  64.],
#                [-28., -64.,  43.,  79.]])
#"""retinanet第2层base_anchors, 最小面积16,109，最大面积41,209"""
#ba2 = np.array([[ -75.,  -29.,  106.,   60.],
#                [ -98.,  -41.,  129.,   72.],
#                [-128.,  -56.,  159.,   87.],
#                [ -48.,  -48.,   79.,   79.],
#                [ -65.,  -65.,   96.,   96.],
#                [ -86.,  -86.,  117.,  117.],
#                [ -29.,  -75.,   60.,  106.],
#                [ -41.,  -98.,   72.,  129.],
#                [ -56., -128.,   87.,  159.]])
#"""retinanet第3层base_anchors, 最小面积65,025，最大面积164,451"""
#ba3 = np.array([[-149.,  -59.,  212.,  122.],
#                [-196.,  -82.,  259.,  145.],
#                [-255., -112.,  318.,  175.],
#                [ -96.,  -96.,  159.,  159.],
#                [-129., -129.,  192.,  192.],
#                [-171., -171.,  234.,  234.],
#                [ -59., -149.,  122.,  212.],
#                [ -82., -196.,  145.,  259.],
#                [-112., -255.,  175.,  318.]])
#"""retinanet第4层base_anchors, 最小面积261,003，最大面积658,377"""
#ba4 = np.array([[-298., -117.,  425.,  244.],
#                [-392., -164.,  519.,  291.],
#                [-511., -223.,  638.,  350.],
#                [-192., -192.,  319.,  319.],
#                [-259., -259.,  386.,  386.],
#                [-342., -342.,  469.,  469.],
#                [-117., -298.,  244.,  425.],
#                [-164., -392.,  291.,  519.],
#                [-223., -511.,  350.,  638.]])
#
#
#def show_bbox(bboxes):
#    """输入array"""
##    x = np.arange(-5.0, 5.0, 0.02)
##    y1 = np.sin(x)
##
##    plt.figure(1)
##    plt.subplot(211)
##    plt.plot(x, y1)
#    wh = [((bb[3]-bb[1]), (bb[2]-bb[0])) for bb in bboxes]
#    areas = [(bb[3]-bb[1])*(bb[2]-bb[0]) for bb in bboxes]
#    print('(w,h) = ', wh)
#    print('areas = ', areas, 'min area = ', min(areas), 'max area = ', max(areas))
#    
#    plt.plot([0,8,8,0,0],[0,0,8,8,0])
#    for bbox in bboxes:
#        plt.plot([bbox[0],bbox[2],bbox[2],bbox[0],bbox[0]],
#                 [bbox[1],bbox[1],bbox[3],bbox[3],bbox[1]])
##        plt.plot((bbox[2],bbox[1]),(bbox[2],bbox[3]))
##        plt.plot((bbox[2],bbox[3]),(bbox[0],bbox[3]))
##        plt.plot((bbox[0],bbox[3]),(bbox[0],bbox[1]))
#
#plt.figure()
#show_bbox(ba0)
#show_bbox(ba1)
#show_bbox(ba2)
#show_bbox(ba3)
#show_bbox(ba4)