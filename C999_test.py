#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:32:50 2019

@author: suliang
"""

import mmcv

out_file = 'test1.json'
outputs = dict(a=1,b=2,c=3)
mmcv.dump(outputs, out_file)  # 先把模型的测试结果输出到文件中