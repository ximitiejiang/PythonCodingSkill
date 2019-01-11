#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 09:49:15 2019

@author: suliang
"""

#----------------1. data-------------------------
'''
Q. tensor的创建
'''
import torch
f1 = torch.tensor([[1,2,3],[4,5,6]])  # 默认float
f2 = torch.tensor(2,3)
f3 = torch.IntTensor([1,2,3])  # 整数tensor

torch.ones(2,3)
torch.zeros(2,3)
torch.eye(2,3)


'''
Q. tensor的数据格式转换
'''
import torch
import numpy as np

# 转成tensor
a = [1,2,3]
b = np.array([3,2,1])
c = dict(a=1,b=2)
torch.tensor(a)  # list转tensor
torch.tensor(b)  # array转tensor
torch.tensor(c)  # dict不可以转tensor

# tensor转其他
t1 = torch.tensor([1,2,3])
t2 = t1.numpy()             # tensor转numpy
t3 = t1.numpy().tolist()    # tensor转numpy,numpy再转list

b0 = torch.tensor(3)
b1 = b0.item()              # 单tensor转标量


'''
Q. tensor的转置跟python不太一样，如何使用，如何避免not contiguous的问题？
1. python 用transpose(m,n,l)可以对3d进行转置，但tensor的transpose(a,b)只能转置2d
   要转置3d需要用permute()
2. 不连续问题
'''
import torch
from numpy import random
# not contiguous的问题：来自pytorch的transpose/permute函数，用切片代替就不会有问题
# 解决方案1：a.contiguous()函数
# 解决方案2：用切片替代transpose/permute
# 解决方案3：用reshape替代view
a0 = torch.tensor(random.randint(1,10,size=(10,10,3)))
a1 = a0.permute(2,0,1)
a1.is_contiguous()  # permute后不连续
a1.view(10,5,6)     # 因为not contiguous报错
a1.contiguous().is_contiguous()
a1.contiguous().view(10,5,6)   # 解决(contiguous函数)
a1.reshape(10,5,6)     # 解决(reshape替代view)

a2 = a0[...,[2,0,1]]   # 解决(用切片代替transpose/permute)
a2.is_contiguous()  # 用切片后是连续的
a3 = a0[...,::-1]   # tensor还不支持负的step
a3.is_contiguous()  # 该操作还不能判断

b0 = torch.tensor(random.randint(1,10,size=(3,4)))
b1 = b0.transpose(1,0)
b1.is_contiguous()  # transpose后不连续





#----------------2. docs(load, save)-------------

#----------------3. other------------------------



