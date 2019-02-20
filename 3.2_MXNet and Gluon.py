# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %%        MXNet基础 
"""验证MXNet安装
1. MXNet需要跟numpy1."""
import numpy as np
import mxnet as mx        # 导入失败，似乎我的numpy版本有问题
a = mx.nd.ones((2,3))     # 

# %%        MXNet基础
"""MXNet支持的数据格式以及特点？
1. MXNet支持的数据格式是NDArray, 缩写成nd，导入为from MXNet import NDArray as nd (这nd跟np很像，跟tensor也很像)
2. 
"""
