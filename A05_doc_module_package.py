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

- 读取模式：r(read), w(write), a(apend), 
       配合b(binary), t(txt), +(read+write)
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

# %%
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
    os.path.isfile(path)：文件是否存在
        
    os.listdir(path)：罗列路径文件夹中所有文件名(非常有用)
    
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

关键理解3： 参考一个独立测试文件test/test_abspath.py
path的语法跟from..import语法有一个地方正好相反：就是从相对根目录的引用地址写法
    >path:      path='./test_data/test.jpg'   用./代表了test/，跟root目录连接上
    >import:    from test.test_data.datasets import VOCDataset，用test.xx直接跟root目录连接上
    这两种连接方式只要反过来就是错的，暂时不知道怎么去理解，就记成：path间接连，import直接连
    
'''
import sys, os

# 获得当前文件相关名称
os.path.abspath(__file__)
os.path.dirname(__file__)
os.path.basename(__file__)

# 获得当前目录父包和父父包路径
os.path.abspath('.')
os.path.abspath('..')


path = './config/config.py'   # 基于当前文件的相对路径
abspath = os.path.abspath(path)       # 相对路径转绝对路径
print(abspath)

relpath = os.path.relpath(abspath)    # 绝对路径转相对路径
print(relpath)

# 在import/from-import语法中：
from test_data.datasets import VOCDataset   # 相对路径：相对于同层以下，永远不会出错
from test.test_data.datasets import VOCDataset
from . test_data.datasets import VOCDataset  # 报错：相对路径：相对于sys.path所加根目录，虽然path语法成功，但from.import失败

# 在path路径语法中：统一用想对于sys.path的相对路径写法，但两种形式./config/config.py, 或 config/config.py
# 但如果__main__文件不在根目录，以下运行
import cv2
import sys, os
import matplotlib.pyplot as plt
path1 = 'test_data/test1.jpg'    # 相对路径写法之1：相对于本层以下的子目录。
print(os.path.abspath(path1))
print(os.path.isfile(path1))
img1 = cv2.imread(path1)
plt.imshow(img1[...,[2,1,0]])

path2 = './test_data/test.jpg'  # 相对路径写法之2：相对于sys.path所加根目录
print(os.path.abspath(path2))
print(os.path.isfile(path2))
img2 = cv2.imread(path2)
plt.imshow(img2[...,[2,1,0]])

path3 = 'test/test_data/test.jpg'  # 报错：相对路径写法之3：相对于sys.path所加根目录
print(os.path.abspath(path3))
print(os.path.isfile(path3))
img3 = cv2.imread(path3)
plt.imshow(img3[...,[2,1,0]])



# %%
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


'''-------------------------------------------------------------------------
Q. 为什么经常导入失败？？？如何通过相对路径，绝对路径导入其他文件夹的模块或者包，有什么区别？
今天在导入data/coco时的错误可能很能说明问题：直接访问'data/coco'报错，改为'../data/coco'就好了。
核心：想要导入一个包或者文件，要到达这个包或者文件，入口永远只有2个，第一个入口是__main__函数文件，
通过这个入口，能够到达main函数的文件树的平级以及下一级的文件并访问，但不可能跳到其他文件树上去。
第二个入口是sys.path中存储的文件夹，通过这个入口同样能够访问该入口文件树的平级以及下级文件，但因为
该入口比较靠根部，这颗文件树比较大所以能够访问的文件也就比较多。

*******这会总结一次完整的关于相对绝对导入的只是********
（默认讨论的都是作为__main__来运行的情况下）
1. import相关的：符合2个入口理论()，且在写相对目录时，因为有隐式相对导入和显式相对导入，所以可不用写父目录地址
    >情况1：导入同一个包里的同级文件，直接用该文件名进行导入即可成功 (但要确保被导入文件内部没有从当前__main__文件再往上去导入，否则报错)
        比如：from B03_dataset_transform import VOCDataset
    >情况2：导入同一个包里的同级文件夹下的文件，直接用该文件夹名进行导入即可成功 (但要确保被导入文件内部没有从当前__main__文件再往上去导入，否则报错)
        比如：from config.config import Config
    >情况3：导入上级文件或其他包里的文件，需要找到共同的父包，然后把这个父包加到sys.path
   然后在该父包之下的所有文件夹/文件都可以直接调用，且不用写父包的名字

2. 文件open相关的：
    同样要满足2个入口理论，唯一区别在于打开文件的目录一定要把父目录包含进去，用./..代替(没有隐式打开目录的方法)

'''
# 情况1
from B03_dataset_transform import VOCDataset
# 情况2
from config.config import Config
# 情况3
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from B.data import b


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
pickle.dump(var_name, f_handle)
var = pickle.load(f_handle)
其中f_handle = open('file_path', 'type'), type可以是r/rb/rt/r+, w/wb/wt/w+, a/ab/at/a+
'''
import pickle
bboxes = [[263, 211, 324, 339], [165, 264, 253, 372], [241, 194, 295, 299]]
# 保存变量
pickle.dump(bboxes, open('bboxes.txt','wb'))
# 加载变量
bb1 = pickle.load(open('bboxes.txt','rb'))


''' --------------------------------------------------------------------------
Q: 如何读取xml文件？
通过ElementTree模块解析xml文件成一棵树：tree = ET.parse(path)
obj.findall('tagname'): 返回list, 是所有tag匹配的element对象的合集
obj.find('tagname'): 返回对象，是第一个tag匹配的element对象
obj.text: 如果obj已经是最后一级tag，则可以用.text命令取出内部数据。

常用操作过程可以简化为如下3步，也就是3种命令，非常简单：
    1. 获得树和根: tree = ET.parse(path)
       获得根: root = tree.getroot()
       由于后续所有操作都是基于根root，所以tree其实可以不用显式获得，以上两句结合成一句即可如下:
       root = ET.parse(path).getroot()
    2. 基于root查找具体对象：
       obj =root.find()         # 查找第一个
       objs = root.findall()    # 查找所有objs并循环调取
       for i in objs:
            ...
    4. 如果已经到最里边一层对象了，则获得字符串
        obj.text
'''
import sys, os
#sys.path.insert(0, os.path.dirname(__file__))
import xml.etree.ElementTree as ET
xml_path = './repo/000001.xml'
tree = ET.parse(xml_path)  # 读取并解析xml文件为一棵树(tree)，为ElementTree对象
root = tree.getroot()      # 找到树根,也是一个Element对象
#访问：均基于root
obj = root.find('object') #返回第一个匹配tag的对象
len(obj)  # 显示有几个tags
objs = root.findall('object')  # 返回所有匹配tag的对象
len(objs)  # 显示有几个obj

bbox=[]
for ob in objs:
    xmin = int(ob.find('bndbox').find('xmin').text)  # 逐层搜索，可写成嵌套方式
    ymin = int(ob.find('bndbox').find('ymin').text)
    xmax = int(ob.find('bndbox').find('xmax').text)
    ymax = int(ob.find('bndbox').find('ymax').text)
    bbox.append([xmin,ymin,xmax,ymax])

n1 = objs[0].find('name')

print(objs[1].find('name').text)
print(obj.find('pose').text)
print(int(objs[0].find('bndbox').find('xmin').text) + 
      int(objs[0].find('bndbox').find('ymin').text))


bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
            


"""-------------------------------------------------------------------------
Q. 如何读写json文件？
json本质上其实是把所有数据转成了str存放，并且转成str过程中对部分数据还做了调整(比如True变成true)
针对json的核心指令主要就是4条：json.load(), json.loads(), json.dump(), json.dumps()
1. json.dumps(data, sort_keys=False, indent=4)是对数据进行编码，形成json格式的数据
    字典转化为json就是一个'dict'，形式上其实一样。可以对key排序，输出显示缩进字符数
2. json.loads(obj, encoding='unicode')是对json数据进行解码，形成python格式的数据
    数据格式对应关系是：
    python(dict/list/str/int/bool/None)分别对应字符串('dict'/'list'/'"str"'/'int'/'true/false'/'null')
3. json.dump(data, f, indent=4)是对数据编码成json格式后存入文件，其中f为打开的文件句柄

4. json.load(f)是从文件中读入json数据并解码为python数据

重要概念： json文件支持的存入数据格式是python的格式，不支持numpy的ndarray，即使嵌套在list里边也不行。
所以如果要存ndarray就需要先把ndarray转换成lsit: data.tolist()

以下是相关子程序(来自mmdetection的JsonHandler)
    def load_from_fileobj(self, file):
        return json.load(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        return json.dumps(obj, **kwargs)
"""
import json
import numpy as np
data = dict(a=1,b=2,c=3,d=4,e=5)
obj = json.dumps(data)
type(obj)

data = [1,2,3]
json.dumps(data)

data = [[1,2],[3,4]]
json.dumps(data)

data = True
json.dumps(data)  # bool经json转化为str：小写的true/false

data = 2.3
json.dumps(data) # 

data = 'Hallo'
json.dumps(data)

data = None
json.dumps(data)

data = np.array([[1,2],[3,4]])
json.dumps(data)           # 报错
json.dumps(data.tolist())  # ok

# 读取和保存到文件
data = dict(a=1,b=2,c=3,d=4,e=5)
with open('test/test_data/test111.json','w') as f:  # 打开文件，如果该文件不存在则先创建
    json.dump(data, f)


# %%
"""----------------------------------------------------------------------
Q. 如何读写pkl文件
pkl文件是利用python的cPickle库支持的一种文件，内容会变成序列化的乱码值。
导入方式是import cPickle as pickle

pickle对比json:
    1. pickle功能更强，可以序列化数据，函数，类等等，但只在python中使用，不被别的认可。且只能以binary(wb/rb)的模式读写
    2. json只能序列化基本数据类型(连numpy都不支持)，但可以在别的数据之间通用转换。是以str的模式读写
    所以多数情况下，用pickle更多也更方便(不用考虑数据格式)，很少用json，除非要跟别的程序做数据交换

注意cPickle, Pickle, six.moves的区别：
1. cPickle是c代码写成，Pickle是python写成，相比之下cPickle更快
2. cPickle只在python2中存在，python3中换成_pickle了
3. six这个包是用来兼容python2/python3的，这应该是six的由来(是2与3的公倍数)
   six包里边集成了有冲突的一些包，所以可以从里边导入cPickle这个在python3已经取消的包

基本命令：
1. pickle.load(f)
2. pickle.dump(f)
3. result = pickle.dumps(data)
4. ori = pickle.loads(result)

    def load_from_fileobj(self, file, **kwargs):
        return pickle.load(file, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        return super(PickleHandler, self).load_from_path(
            filepath, mode='rb', **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('protocol', 2)
        return pickle.dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('protocol', 2)
        pickle.dump(obj, file, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        super(PickleHandler, self).dump_to_path(
            obj, filepath, mode='wb', **kwargs)

"""
from six.moves import cPickle as pickle

# 转换pkl格式变量
data = dict(a=1,b=2)
result = pickle.dumps(data)   # python数据转pkl数据
ori =  pickle.loads(result)   # pkl数据转python数据

# 读取写入pkl文件
data = dict(a=1, b=2)
with open('test_pkl.pkl', 'wb') as f:
    pickle.dump(data, f)   # 必须要用带b的模式写入
    
with open('test_pkl.pkl', 'rb') as f:
    data2 = pickle.load(f)   # 必须要用带b的模式读入

with open('results.pkl', 'rb') as f:
    data3 = pickle.load(f)   # 报错：ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'
                             # 应该是原始pkl文件保存的numpy版本比目标电脑的numpy版本高(mac的是1.14.1)

# 还有pickle的常用用法用来保存变量: 可在命令行直接输入
f = open('test.pkl', 'wb')  # 注意必须用b的模式
pickle.dump(data, f)                             
                             
                             
                             
                             