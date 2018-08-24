#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:23:37 2018

@author: suliang
"""
# below are python skills about the data structures and algorithms

'''
Q: 
'''




'''
Q: 有一个包含N个不同类型元素的序列(list/dic/str)，如何把它分解为N个单独的变量？
'''
p = (4,5)
x,y = p     # 该方法可以同时分解成多个变量

data = ['ACME', 50, 91.1, (2012, 12, 2)]
name, shares, price, (year, mon, day) = data  # 该方法可以嵌套格式分解出多个变量
name, _, price, _ = data   # 该下划线方法可以丢弃掉某些不需要的数值


'''
Q: 有一个字典，怎么对字典里边的值进行计算(最大值，最小值，排序)？
'''
price = {'ACME': 45.23,
         'AAPL': 612.78,
         'IBM': 205.55,
         'HPQ': 37.2,
         'FB': 10.75}
max_price = max(zip(price.values(),price.keys()))  # zip()结合max()求最大最小值
sorted_price = sorted(zip(price.values(),price.keys())) # zip()结合sorted()排序
# zip()创建了一个迭代器，交换key和value, 并且只交换一次，求完max之后就恢复
# max(),sorted()都可以对字典操作，但只能对key操作不能对value操作，所以才要zip()交换
# 该方法优于 max(price.values())，因为能同时返回key和value
# 如果遇到2个最大/小值情况，返回的是两个中键更大/小的那个


'''
Q: ？
'''


