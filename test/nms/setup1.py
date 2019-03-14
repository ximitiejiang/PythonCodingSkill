#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 08:59:57 2019

@author: ubuntu
"""

from distutils.core import setup    # setup是python提供的发布python模块的方法
from Cython.Build import cythonize  # cythonize是cython提供的将python代码转换成c代码的API

"""
1. 创建cython文件.pyx文件
2. 创建setup文件
3. 通过build setup文件，会产生如下结果：(python3 setup1.py build)
    >生成动态链接库(新建build文件夹下产生.so文件: nms_py1.cpython-37m-x86_64-linux-gnu.so)
     在linux是so文件，而在windows是pyd文件
    >创建c代码文件(在同目录下的.c文件: nms_py1.c)，生成的c代码文件比py/pyx文件大了200倍

注意：语法上在setup的最后一行有多一个逗号，跟python不一样
"""
setup(
      name = 'nms_module',
      ext_modules = cythonize('nms_py1.pyx'),
      )