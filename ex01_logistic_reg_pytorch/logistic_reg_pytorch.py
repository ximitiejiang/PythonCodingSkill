#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:58:08 2019

@author: ubuntu
"""

# 逻辑回归分类算法：采用pytorch实现
import numpy as np

class LogisticReg:
    
    def __init__(self, feats, labels, lr=0.001):
        self.n_feats = feats.shape[1]
        self.n_samples = feats.shape[0]
        
        self.feats = np.concatenate([np.ones(self.n_samples, 1), feats])
    
    def train(self):
        
        self.W = np.ones(())
        
if __name__ == "__main__":
    
    logi = LogisticReg()
    