#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 21:22:32 2019

@author: suliang
"""
from torchvision import transforms,Compose
import numpy as np
from numpy import Random

class PytImgTransforms():
    """pytorch transforms
    Args:
        img(PIL img)
    """
    def __init__(self,rotate_cfg):
        self.rotate_cfg = rotate_cfg
    
    def __call__(self, img, hflip=True, rotate=False):
        
        tsfms=[]
        if hflip:
            tsfms.append(transforms.RandomHorizontalFlip(p=0.5))
            
        if rotate:
            degrees = np.array([d*30 for d in range(12)])
            degree = Random.choice(degrees)
            tsfms.append(transforms.RandomRotation(
                degree, resample=False, expand=False, center=True))
        
        tsfms = Compose(np.permutation(tsfms)) # 随机打乱后组合
        img = tsfms(img)   
        return img
    
    