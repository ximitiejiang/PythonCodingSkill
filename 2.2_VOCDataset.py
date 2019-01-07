#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 09:59:03 2019

@author: suliang
0. dataset类最简结构：__getitem__(), __len__()
1. voc数据集基本结构：列表(txt),图片(img),标注(xml)分别在3个不同文件夹
2. 打开并提取txt文件数据
3. 打开并提取xml文件数据
4. 打开img文件
5. 变换img数据和bbox数据

"""
from importlib import import_module
from addict import Dict
import os, sys
import bisect
import numpy as np

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
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
    
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 flip_ratio=0.5,
                 with_label=True,
                 resize_keep_ratio=True):
        
        # 载入txt文件读取图片清单
        ann_file_path = os.path.join(img_prefix, ann_file)
        img_list = []
        with open(ann_file_path) as f:  # 打开txt文件
            lines = f.readlines()   # 一次性读入，分行，每行末尾包含‘\n’做分隔
            for line in lines:
                img_list.append(line.strip('\n'))
                
        #
    
    def load_ann_file(self):
        pass
        
    def __getitem__(self, idx):
        data = self.get_img_and_ann(idx)
        return data
        
    def get_img_and_ann(self, idx):
        pass


class Dataset(object):
    """基础数据集类，增加重载运算__add__对数据集进行叠加
    """
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])
    
class ConcatDataset(Dataset):
    """确定个数的不同来源的数据集堆叠(class from pytorch)，可用于如voc07/12的组合
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)  #得到list(len(p1), len(p1+p2), len(p1+p2+p3),...)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx) #先得到数据集编号
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1] #再得到样本编号
        return self.datasets[dataset_idx][sample_idx]


class RepeatDataset(object):
    """单个数据集的n次循环直到epochs结束
    """
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len


def get_datasets(data_info, parrents, n_repeat=0):
    """用于创建数据集对象：支持多数据集堆叠(ConcatDataset)，支持数据集循环(RepeatDataset)
    Args:
        dset_info(dict): a dict should include{'dset_type', 'ann_file', 
            'img_prefix', 'img_scale', 'img_norm_cfg'}, if ann_file is a list
            means the datasets should concatenate together, and the img_prefix
            should be a list as well.
        n_repeat(int): repeat dataset n times, if 0 means not repeat
        
    Returns:
        obj
    """
    dset_info = Dict(data_info)       # 得到数据集
    dset_type = dset_info.dset_type
    
    dsets = []
    if isinstance(dset_info.ann_file, (list,tuple)):
        assert len(dset_info.ann_file)==len(dset_info.img_prefix)
        for i in len(dset_info.ann_file):
            # 创建类的形参参数字典
            dset_params = Dict()
            dset_params.ann_file = dset_info.ann_file[i]
            dset_params.img_prefix = dset_info.img_prefix[i]
            dset_params.img_scale = dset_info.img_scale
            dset_params.img_norm_cfg = dset_info.img_norm_cfg
            
            dset = obj_generator(parrents, dset_type, dset_params)
            dsets.append(dset)
            
    if n_repeat:
        return RepeatDataset(ConcatDataset(dsets), n_repeat)
    else:
        return ConcatDataset(dsets)
    
    
if __name__ == '__main__':
    # 读入cfg
    path = './repo/voc.py'
    cfg = cfg_from_file(path)
    # 准备数据集信息
    data_info = {}
    data_root = cfg.data_root
    data_cfg = cfg.data
    
    # 
    trainset = get_datasets(data_info, 2)
    
    
    
    
    
    trainset = VOCDataset(data_cfg.train.ann_file, 
                          data_cfg.train.img_prefix,
                          data_cfg.train.img_scale,
                          cfg.img_norm_cfg,
                          size_divisor=16,
                          flip_ratio=0.5)
    data = trainset[0]
        