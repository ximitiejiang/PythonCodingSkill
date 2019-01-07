#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 09:59:03 2019

@author: suliang

1. voc数据集基本结构：列表(txt),图片(img),标注(xml)分别在3个不同文件夹
2. 打开并提取txt文件数据
3. 打开并提取xml文件数据
4. 打开img文件
5. 变换img数据和bbox数据

"""
from importlib import import_module
from addict import Dict
import os, sys
  
def cfg_from_file(path):
    """简版读取cfg函数"""
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


class VOCDataset():
    """简版voc数据集读取类
    """
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 flip_ratio,
                 with_label=True,
                 resize_keep_ratio=True):
        # 载入txt文件
        ann_file_path = os.path.join(img_prefix, ann_file)
        img_list = []
        with open(ann_file) as f:  # 打开txt文件
            data = f.readlines()   # 分行文件相当于‘/n’做分隔
            for line in data:
                img_list.append(line.strip('\n'))
                
            
    
    def load_ann_file(self):
        pass
        
    def __getitem__(self, idx):
        data = self.get_img_and_ann(idx)
        return data
        
    def get_img_and_ann(self, idx):
        pass
    
if __name__ == '__main__':
    #先拿到几个目录地址
    path = './repo/voc.py'
    cfg = cfg_from_file(path)
    data_cfg = cfg.data
    data_root = cfg.data_root
    ann_file = os.path.join(data_root, cfg.data.)
    
    trainset = VOCDataset(data_cfg.train.ann_file, 
                          data_cfg.train.img_prefix,
                          data_cfg.train.img_scale,
                          cfg.img_norm_cfg,
                          size_divisor=16,
                          flip_ratio=0.5)
    data = trainset[0]
        