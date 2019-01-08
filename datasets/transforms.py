#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:28:35 2019

@author: ubuntu

1. 图像变换的逻辑：scale/flip(当前只有水平flip)
2. bbox变换的逻辑：
3. mask变换的逻辑：

"""
__all__ = ['ImageTransforms', 'BboxTransforms']

def img_scale():
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
        
        # normalize
        # bgr2rgb
        # pad to multiple
        # flip
        # transpose
        # to tensor
    
# ---------------------------------------------------------------------------    
class BboxTransforms():
    """bbox变换类"""
    def __init__(self):
        pass
    def __call__(self):
        # scale
        # flip
        # to tensor

# ---------------------------------------------------------------------------
class MaskTransform():
    """mask变换类"""


if __name__=='__main__':
    pass