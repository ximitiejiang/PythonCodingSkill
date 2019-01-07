#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 11:21:57 2019

@author: ubuntu

测试用例，仅提供多个类的module，不包含功能

"""
__all__=['VOCDataset', 'CocoDataset', 'ImagnetDataset', 'TestDataset']

class VOCDataset():
    
    def __init__(self, a=1, b=2):
        print('VOCDataset inited with a={}/b={}!'.format(a,b))

class CocoDataset():
    pass

class ImagnetDataset():
    pass

class TestDataset():
    pass

if __name__ =='__main__':
    d1 = dict(a=3,b=4)
    voc = VOCDataset(**d1)
