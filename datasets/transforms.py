#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:28:35 2019

@author: ubuntu

1. 图像变换的逻辑：
2. bbox变换的逻辑：
3. mask变换的逻辑：

"""
import numpy as np
import cv2
from datasets.color_transforms import bgr2rgb

__all__ = ['ImageTransforms', 'BboxTransforms']


interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

def imresize(img, size, return_scale=False, interpolation='bilinear'):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple): Target (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".

    Returns:
        tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]
    resized_img = cv2.resize(
        img, size, interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale

def imrescale(img, scale, return_scale=False, interpolation='bilinear'):
    """缩放图片：

    Args:
        img (ndarray): The input image.
        scale (float or tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(
                'Invalid scale {}, must be positive.'.format(scale))
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        # min(长边/长，短边/短）确保图片最大程度放大不出界
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            'Scale must be a number or tuple of int, but got {}'.format(
                type(scale)))
    # output (new_w, new_h)    
    new_size = (int(w*scale_factor + 0.5), int(h*scale_factor + 0.5))
    
    rescaled_img = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def imnormalize(img, mean, std):
    img = img.astype(np.float32)  # 为避免uint8与float的计算冲突，在计算类transform都增加类型转换
    return (img - mean) / std
    

def imflip(img, flip_type='h'):
    assert flip_type in ['h','v', 'horizontal', 'vertical']
    if flip_type in ['h', 'horizontal']:
        return np.flip(img, axis=1)
    else:
        return np.flip(img, axis=0)


def impad(img, shape, pad_value=0):
    """图片扩展填充
    Args:
        img(array): img with dimension of (h,w,c)
        shape(list/tuple): size of destination size of img, (h,w) or (h,w,c)
    return:
        padded(array): padded img with dimension of (h,w,c)
    """
    if len(shape) < len(img.shape):
        shape = shape + (img.shape[-1],)
    assert len(shape)==len(img.shape)
    for i in range(len(shape) - 1):
        assert shape[i] >= img.shape[i]
    
    padded = np.empty(shape, dtype = img.dtype)
    padded[...] = pad_value
    padded[:img.shape[0], :img.shape[1], ...] = img
    return padded
    

def impad_to_multiple(img, size_divisor, pad_value=0):
    """图片扩展填充到指定倍数：
    """
    h, w, _ = img.shape
    pad_h = (1 + (h // size_divisor))*size_divisor
    pad_w = (1 + (w // size_divisor))*size_divisor
    return impad(img, (pad_h, pad_w), pad_value)


def bbox_flip(bboxes, img_shape, flip_type='h'):
    """bbox翻转
    Args:
        bboxes(list): [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
        img_shape(tuple): (h, w)
    Returns:
        fliped_img(array): (h,w,c)
    """
    assert flip_type in ['h','v', 'horizontal', 'vertical']
    bboxes=np.array(bboxes)
    h, w = img_shape[0], img_shape[1]
    assert bboxes.shape[-1] == 4
    if flip_type in ['h', 'horizontal']:
        flipped = bboxes.copy()
        # xmin = w-xmax-1, xmax = w-xmin-1
        flipped[...,0] = w - bboxes[..., 2] - 1
        flipped[...,2] = w - bboxes[..., 0] - 1
    else:
        flipped = bboxes.copy()
        flipped[...,1] = h - bboxes[..., 3] - 1
        flipped[...,3] = h - bboxes[..., 1] - 1
        
    return flipped
    

class ImageTransforms():

    def __init__(self, mean, std, to_rgb=True, size_divisor=None):
        """图像变换器初始化
        Args:
            maen(list)
            std(list)
            to_rgb(bool)
            size_divisor(int)
        """
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor
        
    def __call__(self, img, scale, flip=False, keep_ratio=True):
        """图像变换器调用
        Args:
            img((h,w,c)):
            scale(): 
            flip(bool): whether flip horiztal or not
            keep_ratio(bool): whether keep ratio or not when img scale.
        """
        # scale
        if keep_ratio:
            img, scale_factor = imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = imresize(img, scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape
        
        # bgr2rgb，颜色转换务必在normalization之前否则报错，因为颜色转换基于(0~255)
        if self.to_rgb:
            img = bgr2rgb(img)
        # normalize
        img = imnormalize(img, self.mean, self.std)
        # flip
        if flip:
            img = imflip(img, flip_type='h')
        # pad to multiple
        if self.size_divisor:
            img = impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        # transpose
        img = img.transpose(2,0,1) # (h,w,c) to (c,h,w)

        return img, img_shape, pad_shape, scale_factor
    
# ---------------------------------------------------------------------------    
class BboxTransforms():
    """bbox变换类"""
    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts
        
    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
        # scale
        assert isinstance(bboxes, np.ndarray)
        gt_bboxes = bboxes * scale_factor
        # flip
        # TODO: clip?
        gt_bboxes = bbox_flip(gt_bboxes, img_shape, flip_type='h')
        # padding
        if self.max_num_gts:
            num_gts = gt_bboxes.shape[0]
            if num_gts <= self.max_num_gts:
                padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
                padded_bboxes[:self.max_num_gts, :] = gt_bboxes
            else:
                padded_bboxes = gt_bboxes[:self.max_num_gts]
            return padded_bboxes
        else:
            return gt_bboxes
        

# ---------------------------------------------------------------------------
class MaskTransform():
    """mask变换类"""
    pass

# ---------------------------------------------------------------------------
if __name__=='__main__':
    
    id = 2
    
    if id == 1: # 验证ImageTransforms
        path = '../repo/test.jpg'
        img = cv2.imread(path) # (h,w,c)-bgr
        t = ImageTransforms(mean=[123.675, 116.28, 103.53], 
                            std=[1,1,1], 
                            to_rgb=True, 
                            size_divisor=32)
        
        img1, *_ = t(img, scale=(300,300), flip=True, keep_ratio=True) # (c,h,w)
        img2 = img1.transpose(1,2,0)
        
        import matplotlib.pyplot as plt
        plt.subplot(131)
        plt.title('rgb img')
        plt.imshow(img[...,[2,1,0]])
        
        plt.subplot(132)
        plt.title('bgr img')
        plt.imshow(img)
        
        plt.subplot(133)
        plt.title('tsfmed img')
        plt.imshow(img2)
    
    if id == 2: # 验证BboxTransforms
        import pickle
        from visualization.img_show import img_bbox_label_show
        
        path_img = '../repo/test9_img.txt'
        path_bboxes = '../repo/test9_bboxes.txt'
        path_labels = '../repo/test9_labels.txt'
        img = pickle.load(open(path_img,'rb'))  # (h,w,c)-bgr
        bboxes = np.array(pickle.load(open(path_bboxes,'rb')))
        labels = pickle.load(open(path_labels,'rb'))
        
        tsfm1 = ImageTransforms(mean=[123.675, 116.28, 103.53], 
                            std=[1, 1, 1], 
                            to_rgb=True, 
                            size_divisor=32)
        tsfm2 = BboxTransforms()
        
        img1, img_shape, pad_shape, scale_factor = tsfm1(
            img, scale=(300,300), flip=True, keep_ratio=True) # (c,h,w)-rgb
        
        bboxes1 = tsfm2(bboxes, img_shape, scale_factor, flip=True)
        
#        img_bbox_label_show(img, bboxes)
        
        img_bbox_label_show(img1.transpose(1,2,0),bboxes1)
        
        
        
    