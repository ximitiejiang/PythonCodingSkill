#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 09:50:12 2019

@author: suliang

综合知识点：
1. addict库的Dict类
2. importlib库的import_module()函数
3. __dict__的应用, __file__的应用
4. str基本方法
5. sys/os基本方法：os.path.abspath(__file__), 
                 os.path.dirname(path),
                 os.path.basename(path)
6. 列表推导式
"""
from importlib import import_module
from addict import Dict
import os, sys

class Config():
    
    def __init__(self,data):
        """提供另一种生成cfg的方法: cfg = Config(dict(a=1,b=2))
        """
        self.cfg_dict = Dict(data)
    
    def __getattr__(self,name):
        return getattr(self.cfg_dict, name)
    
    @staticmethod    
    def fromfile(path):
        """提供最常用的生成cfg的方法: 兼容dict嵌套模式
        """
        path = os.path.abspath(path)
        if os.path.isfile(path):
            filename = os.path.basename(path)
            dirname = os.path.dirname(path)
            
            sys.path.insert(0,dirname)
            data = import_module(filename[:-3])
            sys.path.pop(0)
            
            _cfg_dict = {name: value for name, value in data.__dict__.items()
                        if not name.startswith('__')}
        return Dict(_cfg_dict)
        
if __name__=='__main__':
    rootpath = os.path.dirname(__file__)
    sys.path.insert(0, rootpath)
    
    path = './repo/ssd300_voc.py'
    cfg = Config.fromfile(path)
    print(cfg.model.backbone.type)
    
    cfg_dict = dict(a=1,b=dict(c=2,d=3))
    cfg2 = Config(cfg_dict)
    print(cfg2.b.d)
            
            
            
        

