#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:01:02 2019

@author: ubuntu
"""

# %%
"""做数据结构和算法分析时用什么计时？
"""
from time import time
start_t = time()
a = 0
for i in range(100000):
    a = (a*2 + 1)/4
last_t = time() - start_t
print('total time: %.4f'%last_t)
