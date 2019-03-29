#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 08:49:47 2019

@author: ubuntu
"""

import numpy as np
import time

#from nms.gpu_nms import gpu_nms   # for gpu 


np.random.seed( 1 )   # keep fixed
num_rois = 6000
minxy = np.random.randint(50,145,size=(num_rois ,2))
maxxy = np.random.randint(150,200,size=(num_rois ,2))
score = 0.8*np.random.random_sample((num_rois ,1))+0.2

boxes_new = np.concatenate((minxy,maxxy,score), axis=1).astype(np.float32)

def nms_test_time1(boxes_new):
    from nms.nms_py import py_cpu_nms  # 这里nms_py是python文件
    thresh = [0.7,0.8,0.9]
    T = 50
    for i in range(len(thresh)):
        since = time.time()
        for t in range(T):

            keep = py_cpu_nms(boxes_new, thresh=thresh[i])     # for cpu
#            keep = gpu_nms(boxes_new, thresh=thresh[i])       # for gpu
        print("thresh={:.1f}, time wastes:{:.4f}".format(thresh[i], (time.time()-since)/T))
    
    return keep

def nms_test_time2(boxes_new):
    from nms.nms_py1 import py_cpu_nms  # 这里nms_py1是c源码文件
    thresh = [0.7,0.8,0.9]
    T = 50
    for i in range(len(thresh)):
        since = time.time()
        for t in range(T):

            keep = py_cpu_nms(boxes_new, thresh=thresh[i])     # for cpu
#            keep = gpu_nms(boxes_new, thresh=thresh[i])       # for gpu
        print("thresh={:.1f}, time wastes:{:.4f}".format(thresh[i], (time.time()-since)/T))
    
    return keep

def nms_test_time3(boxes_new):
    from nms.nms_py2 import py_cpu_nms  # 这里nms_py2是c源码文件,但在cython的pyx文件中优化了变量静态类型
    thresh = [0.7,0.8,0.9]
    T = 50
    for i in range(len(thresh)):
        since = time.time()
        for t in range(T):

            keep = py_cpu_nms(boxes_new, thresh=thresh[i])     # for cpu
#            keep = gpu_nms(boxes_new, thresh=thresh[i])       # for gpu
        print("thresh={:.1f}, time wastes:{:.4f}".format(thresh[i], (time.time()-since)/T))
    
    return keep

def nms_test_time4(boxes_new):
    from nms.gpu_nms import gpu_nms  # 这里nms_py2是c源码文件,但在cython的pyx文件中优化了变量静态类型
    thresh = [0.7,0.8,0.9]
    T = 50
    for i in range(len(thresh)):
        since = time.time()
        for t in range(T):

            keep = gpu_nms(boxes_new, thresh=thresh[i])     # for cpu
#            keep = gpu_nms(boxes_new, thresh=thresh[i])       # for gpu
        print("thresh={:.1f}, time wastes:{:.4f}".format(thresh[i], (time.time()-since)/T))
    
    return keep

if __name__ == "__main__":
    """这个文件主要是为了对比实现nms的几种不同方法在速度上的差异：
    1. python版本的nms
    2. python版本的nms采用cython编译成c版本
    3. python版本的nms直接修改静态变量并存成基于cpu版本的pyx文件，然后采用cython编译成c版本
    4. python版本的nms直接修改静态变量并存成基于gpu版本的pyx文件，然后采用cython编译成c版本
    
    在实际应用中，SSD/M2det算法采用的就是i方式3/4分别编译出nms_cpu/nms_gpu版本的
    """
    ver = 4
    if ver == 1:
        """处理过程：创建py文件，导入调用
        采用python版本nms的时间
        thresh=0.7, time wastes:0.0206
        thresh=0.8, time wastes:0.0785
        thresh=0.9, time wastes:0.3182
        """
        nms_test_time1(boxes_new)
        
    if ver == 2:    
        """处理过程：创建pyx文件，创建setup文件，编译pyx文件，拷贝so文件，导入后调用即可
        采用cython编译后的c版本nms的时间：基本没有提升，有的居然变慢
        (注意需要把so文件拷贝到c文件的同一目录下采用成功导入)
        thresh=0.7, time wastes:0.0221
        thresh=0.8, time wastes:0.0788
        thresh=0.9, time wastes:0.3175
        """
        nms_test_time2(boxes_new)
        
    if ver == 3:
        """处理过程：创建pyx文件，修改pyx中静态变量类型，创建setup文件，编译pyx文件，拷贝so文件，导入后调用即可
        采用cython并进行静态变量类型优化后编译的c版本nms的时间：速度提升惊人！
        (注意需要把so文件拷贝到c文件的同一目录下采用成功导入)
        thresh=0.7, time wastes:0.0012，快了17倍！
        thresh=0.8, time wastes:0.0015，快了52倍！
        thresh=0.9, time wastes:0.0022，快了144倍！！
        """
        nms_test_time3(boxes_new)
        
    if ver == 4:
        """处理过程：创建gpu版本的pyx文件，修改pyx中静态变量类型，
        创建hpp文件给pyx文件调用，
        创建cu文件内核给hpp调用
        然后创建setup文件(指定具体位置的pyx/cu/hpp)，编译pyx文件，拷贝so文件，导入后调用即可
        采用gpu加速版本速度提高跟用静态变量差不多，但如果计算量再大，GPU必然有优势
        thresh=0.7, time wastes:0.0083
        thresh=0.8, time wastes:0.0031
        thresh=0.9, time wastes:0.0042
        """
        nms_test_time4(boxes_new)