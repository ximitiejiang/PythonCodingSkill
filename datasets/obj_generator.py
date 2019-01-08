#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 11:16:33 2019

@author: ubuntu
"""
import sys, os
dirpath = os.path.dirname(__file__)
sys.path.insert(0,dirpath)
from repo import datasets
sys.path.pop(0)


def obj_generator(parrents, obj_type, obj_info=None):
    """generate obj based on class list and specific class assigned.
    Args:
        parrents(module): a module with attribute of all relevant class
        obj_type(str): class name
        obj_info(dict): obj init parameters
    
    Returns:
        obj
    """
    obj_type = getattr(parrents, obj_type)  # 获得类
    if obj_info:
        assert isinstance(obj_info, dict)
        return obj_type(**obj_info)    # 返回带参对象
    else:
        return obj_type()              # 返回不带参对象
    

if __name__ == '__main__':
    dset_type = 'VOCDataset'
    obj_info = dict(a=1, b=2)
    dset = obj_generator(datasets, dset_type, obj_info) # 创建带参对象
    
    dset_type2 = 'CocoDataset'
    dset2 = obj_generator(datasets, dset_type2)  # 创建不带参对象