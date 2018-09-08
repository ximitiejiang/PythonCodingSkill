#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:28:01 2018

@author: suliang
"""

'''
Q: 如何读写文本数据？
'''
with open('xxx.txt', 'rt') as f:  # 用with语句打开文件，只要离开with文件就会自动关闭，省去手动关闭
    data = f.read()               # f.read()一次性全部读入，但读入的是一个字符串

with open(file) as f:            # 
    data = f.readlines()         # f.readlines()一次性全部读入，但读入的是一个list
    for line in data:
        print(line)
        
# 另一种按行读取方式：
with open(file) as f:            # 
    line = f.readline()        # f.readline()一次只读入一行
    while line:
        line = f.readline()
        line = line[:-1]
        line = f.readline()  #再次读入下一行，指针会自动下移，知道读取到最后一行变空退出while

# 写文件
with open('xxx.txt', 'wt') as f:  # 写入模式打开文件
    f.write(text1)               # 写入文件
    
with open('xxx.txt', 'at') as f:  # 以结尾写入的方式打开
    print(text1, file = f)
        

'''
Q: 如何获得文件路径？
'''
# 尽可能用os模块，他能很好处理不同操作系统关于路径的差别，较好的可移植性
import os
path = '~/PythonCodingSkill/4_Iter.py'  # 待获得完整路径的目录名要以～开头
fullpath = os.path.expanduser(path)   # 获得完整路径名


'''
Q: 如何获得某个路径下所有文件名称的列表？
'''
import os
path = '/Users/suliang/MyCodeForML/'
names = os.listdir()
pynames = [name for name in names if name.endswith('.py')]
print(pynames)


'''
Q: 如何读取csv数据，并进行数据汇总和统计？
'''
# python自己建议任何跟数据汇总统计相关的，都用pandas来实现
import pandas as pd
df = pd.read_csv('Train_big_mart_III.csv', skip_footer =1)
df['name'].unique()




