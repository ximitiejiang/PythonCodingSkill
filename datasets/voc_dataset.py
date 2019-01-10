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
__all__ = ['VOCDataset']

from importlib import import_module
from addict import Dict
import os, sys
import bisect
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from transforms import ImageTransforms, BboxTransforms

def obj_generator(parrents, obj_type, obj_info=None):
    """generate obj based on class list and specific class assigned.
    Args:
        parrents(module): a module with attribute of all relevant class
        obj_type(str): class name
        obj_info(dict): obj init parameters
    
    Returns:
        obj
    """
#    obj_type = getattr(parrents, obj_type)
    obj_type = parrents     # 该句微调，直接获得类，因为直接传入了一个class
    if obj_info:
        assert isinstance(obj_info, dict)
        return obj_type(**obj_info)    # 返回带参对象
    else:
        return obj_type()              # 返回不带参对象
  
def cfg_from_file(path):
    """简版读取cfg函数"""
    path = os.path.abspath(path)
    assert os.path.isfile(path)
    
    filename = os.path.basename(path)
    dirname = os.path.dirname(path)
    
    sys.path.insert(0,dirname)
    data = import_module(filename[:-3])
    sys.path.pop(0)
    
    _cfg_dict = {name: value for name, value in data.__dict__.items()
                if not name.startswith('__')}
    return Dict(_cfg_dict)
       

class VOCDataset():
    """简版voc数据集读取类: 内部定义transformer
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
        
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.img_infos = self.load_ann_file(ann_file, img_prefix)
        self.img_scale = img_scale
        self.img_norm_cfg = img_norm_cfg
        
        self.img_transforms = ImageTransforms(**self.img_norm_cfg)
        self.bbox_transforms = BboxTransforms()
        self.cat2label = {key: i for i, key in enumerate(self.CLASSES)}
    
    def load_ann_file(self, ann_file, img_prefix):
        """同时读取txt文件和xml文件，组成图片变量img_infos
        Args:
            ann_file(str)
        Returns:
            img_infos(list(dict)): include [{'img_id','img_path','width','height'}]
        """
        # 载入txt文件读取图片id
        img_ids = []
        with open(ann_file) as f:  # 打开txt文件
            lines = f.readlines()   # 一次性读入，分行，每行末尾包含‘\n’做分隔
            for line in lines:
                img_ids.append(line.strip('\n'))        
        # 读取xml文件中的width/height组成每张图片专属变量img_infos
        img_infos = []
        for img_id in img_ids:
            img_path = img_prefix + 'JPEGImages/' + '{}.jpg'.format(img_id)
            xml_path = img_prefix + 'Annotations/' + '{}.xml'.format(img_id)
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                Dict(img_id=img_id, img_path=img_path, xml_path=xml_path, width=width, height=height))    
        return img_infos
            
    def __len__(self):
        return len(self.img_infos)
    
    def __getitem__(self, idx):
        data = self.prepare_img_and_ann(self.img_infos, idx)
        return data
     
    def prepare_img_and_ann(self, img_infos, idx):
        # 读取单张图片
        img = cv2.imread(img_infos[0]['img_path']) # (h,w,c) - bgr
        # 读取单个xml中的bbox数据
        tree = ET.parse(img_infos[0].xml_path)
        root = tree.getroot()
        bboxes = []         # 存放difficult =0的数据
        labels = []  
        bboxes_ignore = []  # 存放difficult =1的数据
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            if difficult:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes: # 如果没有difficult=0的数据，则创建空数组
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore: # 如果没有difficult=1的数据，则创建空数组
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        
        # img transform: scale/2rgb/norm/flip/padding/transpose -> (c,h,w)-rgb
        img, img_shape, pad_shape, scale_factor = self.img_transforms(
            img, scale=self.img_scale, flip=True, keep_ratio=True) 
        # bbox transform: scale/flip/num_filter_if_need 
        bboxes = self.bbox_transforms(ann['bboxes'],img_shape, scale_factor) #
        
        # debug
        from visualization.img_show import img_bbox_label_show
        img1 = img.transpose(1,2,0)
        img_bbox_label_show(img1,bboxes)
        
        # 生成img_meta
        img_meta = dict()
        # 生成data
        data = dict(img = img, img_meta = img_meta, ann = ann)
        
        return data
        

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
        """把ConcatDataset的len函数设计成list,包含每个dataset的长度累加值，
        更便于频繁操作__getitem__函数的编写(__getitem__要尽可能高效)
        """
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


def get_datasets(dset_info, parrents, n_repeat=0):
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
    dset_info = dset_info.dataset.copy()
    dset_type = dset_info.type
    
    dsets = []
    if isinstance(dset_info.ann_file, (list,tuple)):
        assert len(dset_info.ann_file)==len(dset_info.img_prefix)
        for i in range(len(dset_info.ann_file)):
            # 创建类的形参参数字典：该字典还可以加入其他参数
            dset_params = Dict()
            dset_params.ann_file = dset_info.ann_file[i]
            dset_params.img_prefix = dset_info.img_prefix[i]
            dset_params.img_scale = dset_info.img_scale
            dset_params.img_norm_cfg = dset_info.img_norm_cfg
            if dset_params.get('size_divisor'):
                dset_params.size_divisor = dset_info.size_divisor
            if dset_params.get('flip_ratio'):
                dset_params.flip_ratio = dset_info.flip_ratio
                
            dset = obj_generator(parrents, dset_type, dset_params)
            dsets.append(dset)
            
    if n_repeat:
        return RepeatDataset(ConcatDataset(dsets), n_repeat)
    else:
        return ConcatDataset(dsets)
    
    
if __name__ == '__main__':
    # 读入cfg
    path = '../repo/voc.py'
    cfg = cfg_from_file(path)
    
    # 
    trainset = get_datasets(cfg.data.train, VOCDataset, 0)
    
    #TODO: transforms to be finished
    data = trainset[3000]  # idx=6000(属于voc2012)的图片有点奇怪
        