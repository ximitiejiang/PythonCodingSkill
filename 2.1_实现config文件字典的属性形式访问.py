#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 09:50:12 2019

@author: suliang

综合知识点：
1. addict库的Dict类
2. importlib库的import_module()函数: 
    缺少基础实例？？？
    可导入一个moduel文件，所有变量作为属性，需要通过__dict__调出来
    只接收不带扩展名的文件名(不带.py)
    
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
    """最简单的实现，是直接定义Config继承Dict即可
    但为了让Config能够更rubust，更多基础功能，所以接口统一成输入普通dict
    在Config内部自定义转换成Dict，并定义__getattr__和__getitem__
    """
    def __init__(self,data):
        self.cfg_dict = Dict(data)
   
    def __getattr__(self,name):
        return getattr(self.cfg_dict, name)  # 调用Dict类自己的__getattr__来支持嵌套
    
    def __len__(self):
        return len(self.cfg_dict)
    
    @staticmethod    
    def fromfile(path):
        """从文件提取一个dict，送入Config
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
        return Config(_cfg_dict)

# 方案2,一个更简单的Config, 直接定义成Dict，但功能不能更多定制
class Config_2(Dict):    
    @staticmethod    
    def fromfile(path):
        """从文件提取一个dict，送入Config
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
        return Config(_cfg_dict)


if __name__=='__main__':
    rootpath = os.path.dirname(__file__)
    sys.path.insert(0, rootpath)
    # 方式1
    path = './repo/voc.py'
    cfg = Config.fromfile(path)
    cfg1 = cfg.model.backbone.type
    # 方式2
    cfg_dict = dict(a=1,b=dict(c=2,d=3))
    cfg2 = Config(cfg_dict)
    cfg3 = cfg2.b.d
            
            
            
        

