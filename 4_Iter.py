#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:27:22 2018

@author: suliang
"""

# 迭代器和生成器

'''
Q: 想要迭代一个序列，但想同时记录下迭代序列当前的元素索引，怎么做？
'''
mylist = ['a', 'b', 'c']
for id, value in enumerate(mylist): # 使用内置enumerate()函数同时获得序列的索引和数值
    print(id, value)
    
    
'''
Q: 想要同时迭代多个序列，怎么做？
'''
listx = [1,5,4,2,10,7]
listy = [101,78,37,15,62,99]
# 使用内置的zip(a, b)函数构建一个迭代器元祖(a, b)
# zip()也可以接受2个以上的序列组成迭代器zip(a, b, c)
for x, y in zip(listx, listy):  
    print(x, y)
    
    
'''
Q: 同时迭代多个序列，还可以用字典实现吗？
'''    
user = {'Eason':['toyota'],
        'Leo':['ford','honda','smart'],   # 字典可以嵌套列表作为值
        'Winnie':['nissan','nio']}
for name, cars in user.items():      # 遍历键值对：用.items()返回的是一个键值对元组列表
    for car in cars:                 # 对嵌套列表值进行二级循环遍历
        print(name + ': '+ car)
        

'''
Q: 同时？
'''            






        