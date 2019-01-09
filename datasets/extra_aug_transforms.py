#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 08:21:00 2019

@author: ubuntu
"""
import numpy as np
from numpy import random
from color_transforms import bgr2hsv, hsv2bgr
from models.head_support import bbox_overlaps    

class PhotoMetricDistortion(object):
    """随机调整图像的brightness/contrast/saturation/hue/
    随机0不调整，1调整，调整值也是随机从调整范围中获取
    brightness: img(bgr) + delta
    contrast: img(bgr)*alpha
    saturation: img(hsv)[...,1]*alpha
    hue: img(hsv)[...,0]+delta
         img(hsv)
    swap channel: img(bgr)[...,random.permutation(3)]
    
    """
    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2): 
            # need float32 as input, not uint8(preprocess done in 
            # ExtraAugmentation class __init__: img = img.astype(np.float32))
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]  # range(3), bgr(h,w,c)变成brg/grb/rgb等形式

        return img, boxes, labels


class Expand(object):
    """随机放大图片n倍，然后把原图随机放到大图某位置
    """
    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        
        boxes += np.tile((left, top), 2)
        return img, boxes, labels


class RandomCrop(object):
    """随机切割
    """
    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, labels):
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode) # 先从[1, min_ious, 0]之间随机选个数
            if mode == 1:
                return img, boxes, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w) # (0.3w, w)
                new_h = random.uniform(self.min_crop_size * h, h) # (0.3h, h)

                # h / w in [0.5, 2] 避免太狭长的图片出现
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                return img, boxes, labels


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))

    def __call__(self, img, boxes, labels):
        # img from cv2.imread is uint8, need transfer to float32, 
        # to calculate with float32 when augmentation
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels
    
if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    path = '../repo/test.jpg'
    img = cv2.imread(path) # (h,w,c)-bgr
    img = img.astype(np.float32)
    bbox = np.array([[-14,-14,20,20]])
    label = np.array([[1]])
    
    id = 3
    
    if id == 2: # 验证2：expand效果
        t2 = Expand(
            mean=[123.675, 116.28, 103.53], 
            to_rgb=True, ratio_range=(1, 4))
        img1, *_ = t2(img,bbox,label)
        img2, *_ = t2(img,bbox,label)
        img3, *_ = t2(img,bbox,label)
        
        plt.subplot(141)
        plt.imshow(img.astype(np.int))
        plt.subplot(142)
        plt.imshow(img1.astype(np.int))
        plt.subplot(143)
        plt.imshow(img2.astype(np.int))
        plt.subplot(144)
        plt.imshow(img3.astype(np.int))
    
    
    if id == 1: # 验证1：distortion效果     
        t1 = PhotoMetricDistortion(
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18)
        img1, *_ = t1(img,bbox,label)
        img2, *_ = t1(img,bbox,label)
        img3, *_ = t1(img,bbox,label)
        plt.subplot(131)
        plt.imshow(img1.astype(np.int))
        plt.subplot(132)
        plt.imshow(img2.astype(np.int))
        plt.subplot(133)
        plt.imshow(img3.astype(np.int))
    
    if id == 3: # 验证3： crop，没有看到变化，待分析函数逻辑
        t3 = RandomCrop(
            min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
            min_crop_size=0.3)
        img1, *_ = t3(img,bbox,label)
        img2, *_ = t3(img,bbox,label)
        img3, *_ = t3(img,bbox,label)
        plt.subplot(221)
        plt.imshow(img.astype(np.int))
        plt.subplot(222)
        plt.imshow(img1.astype(np.int))
        plt.subplot(223)
        plt.imshow(img2.astype(np.int))
        plt.subplot(224)
        plt.imshow(img3.astype(np.int))
    
    