#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 08:52:04 2019

@author: ubuntu
"""
"""总结1：对于from..import/import这类导入的相对路径，要么是基于根目录下的相对路径
错误写法：from .datasets.class_names xxx，这种写法.相当与重写了一边sys.path的根目录造成重复
错误写法：from .test_data.datasets xxx, 这种写法


path的语法跟from..import语法有一个地方正好相反：就是从相对根目录的引用地址写法
    >path:      path='./test_data/test.jpg'   用./代表了test/，跟root目录连接上
    >import:    from test.test_data.datasets import VOCDataset，用test.xx直接跟root目录连接上
    这两种连接方式只要反过来就是错的，暂时不知道怎么去理解，就记成：path间接连，import直接连
    
"""
from test_data.datasets import VOCDataset   # 相对路径：相对于同层以下
#from . test_data.datasets import VOCDataset  # 报错：相对路径：相对于sys.path所加根目录，虽然path语法成功，但from.import失败
from test.test_data.datasets import VOCDataset # 相对路径：相对于sys.path根目录
import cv2
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    """总结1：对于path中的相对路径，要么是在根目录下的相对路径，要么是在同层以下相对路径
    """  
    path1 = 'test_data/test1.jpg'    # 相对路径写法之1：相对于本层以下的子目录。
    print(os.path.abspath(path1))
    print(os.path.isfile(path1))
    img1 = cv2.imread(path1)
    plt.imshow(img1[...,[2,1,0]])

    path2 = './test_data/test.jpg'  # 相对路径写法之2：相对于sys.path所加根目录
    print(os.path.abspath(path2))
    print(os.path.isfile(path2))
    img2 = cv2.imread(path2)
    plt.imshow(img2[...,[2,1,0]])
    
    path3 = 'test/test_data/test.jpg'  # 报错：相对路径写法之2：相对于sys.path所加根目录
    print(os.path.abspath(path3))
    print(os.path.isfile(path3))
    img3 = cv2.imread(path3)
    plt.imshow(img3[...,[2,1,0]])
    
    

    