#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:28:35 2019

@author: ubuntu
"""

class ImageTransforms():
    """图片变换
    Args:
        maen(list)
        std(list)
        to_rgb(bool)
        size_divisor(int)
    """
    def __init__(self, mean, std,to_rgb=True, size_divisor=None):
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor
        
    def __call__(self, imgs, flip):
        pass
    
    
class BboxTransforms():
    """bbox变换类"""
    def __init__(self):
        pass
    def __call__(self):
        pass


if __name__=='__main__':
    pass