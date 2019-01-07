#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:28:01 2018

@author: suliang
"""

''' --------------------------------------------------------------------------
Q: 如何读取文本数据？

with open() as f 是一个良好的打开文档的习惯，会自动关闭文档
- f为一个迭代对象, 可以用在for循环和next()两种语句中，
  同时可以有如下3种操作方法：
    - f.read()  读取全部数据
    - f.read(size)
    - f.readline() 读取1行
    - f.readlines() 读入所有行

- 读取模式：r(read), w(write), a(apend), 配合b(binary), t(txt), +(read+write)
    r: 只能读，文件不存在就报错
    r+: 可读可写，文件不存在就报错，覆盖模式
    
    w: 只能写，文件不存在就创建，覆盖模式
    w+: 可读可写，文件不存在就创建，覆盖模式  (*)  - 可理解为覆盖write的自由模式
    a: 只能写，文件不存在就创建，追加模式
    a+: 可读可写，文件不存在就创建，追加模式  (*)  - 可理解为apend的自由模式
'''
with open('test.txt', 'rt') as f:  # 用with语句打开文件，只要离开with文件就会自动关闭，省去手动关闭
    a = next(f)
    print(a)
    data = f.read()               # f.read()一次性全部读入，但读入的是一个字符串, 还需调用字符串公式进行分割

with open('test.txt', 'rt') as f:            # 
    data = f.readlines()         # f.readlines()一次性全部读入，但读入的是一个list, 每行一个字符串
    for line in data:
        print(line)
        
# 另一种按行读取方式：
with open(file) as f:            # 
    line = f.readline()        # f.readline()一次只读入一行
    while line:
        line = f.readline()
        line = line[:-1]
        line = f.readline()  #再次读入下一行，指针会自动下移，知道读取到最后一行变空退出while


''' --------------------------------------------------------------------------
Q: 如何写文件？
- 跟读取文件是一样的，只是模式用w, wb分别代表写入文本文件或二进制文件
- 跟读取文件的模式也是一样的
'''
with open('xxx.txt', 'wt') as f:  # 写入模式打开文件
    f.write(text1)               # 写入文件
    
with open('xxx.txt', 'at') as f:  # 以结尾写入的方式打开，只有'a'和'at'两种模式的指针是在文件末尾
    print(text1, file = f)

def write_txt(results,file_name):
    '''用于写入一个csv文件，默认路径在checkpoints文件夹
    '''
#    import torch
#    import numpy as np
#    results = torch.tensor([1,2,3])  # tensor写入形式为'tensor([1,2,3])'
#    results = 'hello'                # 字符串写入形式为string
#    results = [1,2,3]                # list的写入形式为[1,2,3]
#    results = np.array([3,1,2])      # array的写入形式为[1,2,3]
    with open('/Users/suliang/slcv/checkpoints/111.txt', 'w+') as f:  # 以结尾写入的方式打开，只有'a'和'at'两种模式的指针是在文件末尾
        print(results, file = f)

    
''' --------------------------------------------------------------------------
Q: 如何通过print语句把待输出内容写入文件中？
'''
with open('xx.txt', 'rt') as f:
    print('Hello World!', file = f)  # 加上file关键字即可


''' --------------------------------------------------------------------------
Q: 如何获得文件绝对路径？
'''
# 尽可能用os模块，他能很好处理不同操作系统关于路径的差别，较好的可移植性
import os
path = '~/PythonCodingSkill/4_Iter.py'  # 待获得完整路径的目录名要以～开头
fullpath = os.path.expanduser(path)   # 获得完整路径名
print(fullpath)


''' --------------------------------------------------------------------------
Q: 如何把相对路径转化为绝对路径？又如何把绝对路径转化为相对路径
关键理解1：
    os.path.abspath(path)：获得绝对路径，等效于增加当前main的路径
    os.path.basename(path)：获得文件名
    os.path.dirname(path)：获得文件路径
    dir, name = os.path.split(path)：获得路径和文件名
    os.path.join(dir,base)：拼接
    os.path.expanduser(path)：替换user为实际路径
    os.path.exists(path)：路径是否存在
    os.isfile(path)：文件是否存在
    sys.path.insert(0, path)
    sys.path.pop(0)

关键理解1： os.path.abspath只是简单的获得指定文件的绝对路径，而不会关心文件是否存在
或者可以理解为只是简单的把.和..替换成真实路径，然后拼接后边一段。

关键理解2：'.'和'..'简化理解就是文件夹，并且是不带/的文件夹名。相对路径几种写法
需要区分文件夹和模块的写法区别，文件夹要带斜杠，模块不带斜杠带.分隔
    path = './slcv/slcv/cfg/config.py'
    path = '../slcv/slcv/cfg/config.py'
    from .config import Config
    from . import config
    from ..slcv.config import Config
    
'''
import os
path = '../slcv/slcv/cfg/config.py'   # 基于当前文件的相对路径
abspath = os.path.abspath(path)       # 相对路径转绝对路径
print(abspath)

relpath = os.path.relpath(abspath)    # 绝对路径转相对路径
print(relpath)

''' --------------------------------------------------------------------------
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
os.path.join(dir, base)        # 拼接文件名: 
# 生成每个文件的绝对地址
name_addr = [os.path.join(root, name) for name in names]  # 拼接地址
print(name_addr)
"""注意 os.path.join(addr1, addr2, addr3)的用法:
(1)只认最后一个以/开始的根目录以及之后的拼接目录，该根目录以左的所有目录都会被忽略
其中根目录是指以/开始的目录
(2)拼接时结尾的/可有可无，命令会自动调整成正确形式
"""
dir1 = os.path.join('/aa/bb/c','/d/e/','f/g/h')
dir2 = os.path.join('/aa/bb/c','/d/e', 'f/g/h')
print(dir1)  # 只会从第二个/d/e/开始算起
print(dir2)


''' --------------------------------------------------------------------------
Q: 如何读取csv数据，并进行数据汇总和统计？
'''
# python自己建议任何跟数据汇总统计相关的，都用pandas来实现
import pandas as pd
df = pd.read_csv('Train_big_mart_III.csv', skip_footer =1)
df['name'].unique()


''' --------------------------------------------------------------------------
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


''' --------------------------------------------------------------------------
Q: 如何读取其他文件夹的包？
'''
from KidsCoding.kc.tool import PAPER,DRAW,ANT
# 只要在KidsCoding包，kc包都存在时，从tool文件中导入类就是合法的

paper = PAPER(size=[6,8])
ant = ANT(paper)

ant.move(2)



''' --------------------------------------------------------------------------
Q: 如何一次性导入多个类的集合？
1. 最好一级管一级，每一层导入管到自己下一级就可以了，避免太宽管不过来。
2. 每集定义好__all__接口，也是一个好的编程习惯

'''
# models作为一个package包，在他的__init__文件中
#先导入所有类，然后打包成__all__变量
from .base import BaseDetector
from .single_stage import SingleStageDetector
from .two_stage import TwoStageDetector
from .rpn import RPN
from .fast_rcnn import FastRCNN
from .faster_rcnn import FasterRCNN
from .mask_rcnn import MaskRCNN
from .cascade_rcnn import CascadeRCNN
from .retinanet import RetinaNet

__all__ = [
    'BaseDetector', 'SingleStageDetector', 'TwoStageDetector', 'RPN',
    'FastRCNN', 'FasterRCNN', 'MaskRCNN', 'CascadeRCNN', 'RetinaNet'
]

# 然后在其他文件中引用



''' --------------------------------------------------------------------------
Q: 如何保存变量和加载变量
'''
import pickle
# 保存变量
pickle.dump('', path)
# 加载变量
pickle.load(path)


''' --------------------------------------------------------------------------
Q: 如何读取xml文件

'''