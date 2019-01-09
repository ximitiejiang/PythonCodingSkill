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
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            'Scale must be a number or tuple of int, but got {}'.format(
                type(scale)))
    new_size = _scale_size((w, h), scale_factor)
    rescaled_img = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img

def imnormalize(img, mean, std):
    pass
    


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
        # normalize
        img = imnormalize(img, self.mean, self.std, self.to_rgb)
        # bgr2rgb
        img = bgr2rgb(img)
        # pad to multiple
        
        # flip
        
        # transpose
        img = img.transpose(2,0,1) # (h,w,c) to (c,h,w)
        # to tensor
        return img
    
# ---------------------------------------------------------------------------    
class BboxTransforms():
    """bbox变换类"""
    def __init__(elf, max_num_gts=None):
        pass
    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
        # scale
        
        # flip
        
        # to tensor
        
        return bboxes

# ---------------------------------------------------------------------------
class MaskTransform():
    """mask变换类"""
    pass


if __name__=='__main__':
    pass