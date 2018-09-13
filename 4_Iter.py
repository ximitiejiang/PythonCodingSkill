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
Q: 想要迭代一个字典，但想同时迭代字典的键和值，怎么做？
'''
# 同时循环键 + 值： for key, value in dic.items()
# 只循环键： for key in dic.keys()
# 只循环值： for value in dic.values()
mydict = {'alen':3, 'backer':2, 'cathy':4,'leo':9, 'eason':20}
for key, value in mydict.items():# 使用.items()方法同时获得键和值
    print(key, value)

for key in mydict.keys():
    print(key)
    
for value in mydict.values():
    print(value)

    
    
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
Q: 想要？
'''    