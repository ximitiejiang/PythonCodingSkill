#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:26:13 2018

@author: suliang
"""

'''
Q: 如何生成一组随机数？
'''
# 使用random模块，一般用于产生1个随机数
import random
import numpy as np
a = random.randint(1,10)  # 产生一个随机整数
b = random.random(1,10)  # 产生一个随机浮点数

# 使用numpy.random模块，一般用于产生随机矩阵
c = np.random.randn(2,2) # 产生随机矩阵




'''
Q: 如何从一组已知数中随机挑选一个数？
'''
import random
values = [1,2,3,4,5,6]
random.shuffle(values)   # 随机打乱一个list,只有inplace方式

a = random.choice(values)  # 随机取出1个数
b = random.sample(values, 3)  # 随机取出多个数


'''
Q: 如何让每次产生的随机数是一样的？
'''
import numpy as np
import random
random.seed(1)
a = np.random.randn(2,2)


