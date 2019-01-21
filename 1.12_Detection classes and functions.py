#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:24:32 2019

@author: ubuntu
"""
# %%
'''Q. 对比SingleStageDetector与TwostageDetector是如何抽象出来的？
1. 共用的部分(抽象出来的部分)：
2. 独立的部分(重写的部分)：
3. 
'''
# 先看BaseDetector: 
class BaseDetector(nn.Module):
    """base detector作为一个基类来定义的"""
    __metaclass__ = ABCMeta
    def __init__(self):
        super().__init__()
    @abstractmethod
    def extract_feat(self, imgs):
        """这是强制需要定义的方法"""
        pass
    def forward_train():
        pass
    
# 再看single stage detector
class SingleStageDetector:
    """singlestage detector用于给ssd/yolo"""
    def forward_train():
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list
        bbox_results
        


# %%
'''
'''    