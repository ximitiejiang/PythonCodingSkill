#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:55:26 2019

@author: ubuntu
"""
import os
import cv2
import torch
from torch.utils.data import Dataset
from transforms import ImageTransforms, to_tensor
"""继承pytorch自带的Dataset类，包含3个方法，__getitem__/__len__是必须要重写的方法
还自带一个__add__方法，用于多数据源堆叠
"""

class DogcatDataset(Dataset):
    """猫狗数据集比较简单：只有两个train/test图片包, 每个图片包通过文件名
    识别label，带cat的文件名是猫，带dog的文件名是狗
    """
    Classes = ['dog', 'cat']
    
    def __init__(self, root, transform=None, train=True, test=False):
        self.transform = transform
        if not self.transform:
            self.transform = ImageTransforms(mean=[0.5,0.5,0.5], std=[0,0,0], to_rgb=True)
        if train:
            root = os.path.join(root, 'train')
            self.imgs = []
            self.labels = []
            for img_name in os.listdir(root):
                self.imgs.append(os.path.join(root, img_name))
                self.labels.append(img_name.split('.')[0])
        else:
            root = os.path.join(root, 'test')
            self.imgs = []
            self.labels = []
            for img_name in os.listdir(root):
                self.imgs.append(os.path.join(root, img_name))
                self.labels.append(img_name.split('.')[0])
    
    def __getitem__(self, idx):
        """return img, label for trainset, return img, num_id for testset
        default transform: scale, flip, to_rgb, normalize, transpose_to_chw
        
        Returns:
            img(array): (c,h,w)
            label(str): 
        """
        img_path = self.imgs[idx]
        img = cv2.imread(img_path)  # (h,w,c)-bgr
        img, *_ = self.transform(img, scale=1, flip =True)
        label = self.labels[idx]
        
        img = to_tensor(img)
        label = to_tensor(label)
        
        return (img, label)
    
    def __len__(self):
        return len(self.imgs)


if __name__=='__main__':
    
    # 数据集简单读取显示
    root = '../data/DogsCats'
    dc = DogcatDataset(root, train=False, test=False)
    print(len(dc))
    img1, label1 = dc[5]
    import matplotlib.pyplot as plt
    plt.imshow(img1[...,[2,1,0]])  # bgr转rgb显示
    print(label1)   # 作为测试集只显示id
    
    # 数据集堆叠实验
    dc1 = DogcatDataset(root)
    dc2 = DogcatDataset(root)
    dc3 = dc1 + dc2
    print(len(dc3))
    img2, label2 = dc3[30000]
    import matplotlib.pyplot as plt
    plt.imshow(img2[...,[2,1,0]])  # bgr转rgb显示
    print(label2)   # 训练集显示类型
    