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
    
with open('xxx.txt', 'at') as f:  # 以结尾写入的方式打开，只有'a'和'at'两种模式的指针是在文件末尾
    print(text1, file = f)
        


'''
Q: 如何获得文件绝对路径？
'''
# 尽可能用os模块，他能很好处理不同操作系统关于路径的差别，较好的可移植性
import os
path = '~/PythonCodingSkill/4_Iter.py'  # 待获得完整路径的目录名要以～开头
fullpath = os.path.expanduser(path)   # 获得完整路径名
print(fullpath)


'''
Q: 如何获得某个路径下所有文件名称的列表？
'''
import os
root = '/Users/suliang/PythonCodingSkill/'
names = os.listdir(root)  # 文件名称列表
pynames = [name for name in names if name.endswith('.py')]
print(pynames)

root = '/Users/suliang/PythonCodingSkill/abc.py'
base = os.path.basename(root)   # 获得文件名
dir = os.path.dirname(root)    # 获得文件路径
os.path.join(dir, base)        # 拼接文件名

# 生成每个文件的绝对地址
name_addr = [os.path.join(root, name) for name in names]  # 拼接地址
print(name_addr)


'''
Q: 如何读取csv数据，并进行数据汇总和统计？
'''
# python自己建议任何跟数据汇总统计相关的，都用pandas来实现
import pandas as pd
df = pd.read_csv('Train_big_mart_III.csv', skip_footer =1)
df['name'].unique()


'''
Q: 如何设置文件的包(package)和模块(module)，并进行模块导入
假定如下文件结构：
mypackage/
    __init__.py
    packA/
        __init__.py
        spam.py
        grok.py: class AAA()
    packB/
        __init__.py
        bar.py: class BBB()

需要理解包与模块的概念：包是package对应文件夹，模块是module对应py文件，
通常一个包里边包含多个模块，相当与一个文件夹包含多个py文件
__init__文件可以把文件夹变成一个包/package，这样这个文件夹(package)就可以被import导入了
import命令，可以导入一个package, 导入一个module，或者导入一个.py文件
'''
# 需要理解包package是文件夹, 模块module是文件, 类class是代码段
# 首先要把文件夹转化为包package, 即在文件夹下创建__init__.py文件

# 如果在spam.py文件中希望导入grok模块, 可以用相对导入.grok
from .grok import AAA  # 方式1：导入AAA类, .代表同级目录
from . import grok     # 方式2：导入grok模块
import packA           # 方式3：只导入packA, 然后在packA下面init文件中添加 from .grok import AAA
# 但由于相对导入实现前提是父目录已经导入，也就是packA要先导入，否则以下语句会报错：
# ModuleNotFoundError: No module named '__main__.Basicmodule'; '__main__' is not a package
# 所以合适的写法是如下：
from packA.grok import AAA

# 如果在spam.py文件中希望导入bar模块
from ..packB import bar     # 方式1：导入bar模块, ..代表上一级目录
from ..packB.bar import BBB # 方式2：导入BBB类
import packB                # 在packB下面init文件中添加from .bar import BBB



'''
Q: 如何读取其他文件夹的包？
'''
from KidsCoding.kc.tool import PAPER,DRAW,ANT
# 只要在KidsCoding包，kc包都存在时，从tool文件中导入类就是合法的

paper = PAPER(size=[6,8])
ant = ANT(paper)

ant.move(2)
