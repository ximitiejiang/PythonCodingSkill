#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 22:35:26 2019

@author: ubuntu
"""
import sys

def add(arg1, arg2):
    return arg1 + arg2

if __name__ == '__main__':
    # 注意：sys.argv返回的是一个list, 里边每个元素都是str字符串。
    # 所以要通过切片提取参数，同时要把字符串转换成int
    result = add(int(sys.argv[1]), int(sys.argv[2]))  
    print(result)
    
    """
    调用方法：
    python ./test2.py 12 24  
    """
    
