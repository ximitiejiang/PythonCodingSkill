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
"""Q. 把路径加到sys.path的最简单方法
"""
import sys, os

sys.path.insert(os.path.abspath('..')) # 把当前所在文件夹的父文件夹加入sys.path

sys.path.insert(os.pardir)        # 等效的方法，更简洁。
    
    

# %%
"""Q. 如果两个module文件产生交叉导入，或者叫交叉引用，导致报错无法导入如何解决？
参考1：https://blog.csdn.net/qq_34146899/article/details/52530844 （提供的方法解决了我的问题）
参考2：https://blog.csdn.net/polyhedronx/article/details/81911580 （提供了背后的机理）

比如：
在utils/prepare_training这个module文件中，导入了models/detector_lib这个module文件中的类OneStageDetector
在models/detector_lib这个module文件中，又反过来导入了utils/prepare_training这个module文件中的get_model这个函数。
两个module文件相互引用对方，会导致报错。

解决方案1：在models/detector_lib中把import get_model这句话放到具体使用的位置的前一句去，而不是放在文件最开始，即可导入。

"""




# %%
'''Q: 如何把相对路径转化为绝对路径？又如何把绝对路径转化为相对路径
关键理解1：
    os.pardir: 常量，代表'..'
    os.curdir: 常量，代表'.'
    os.path.abspath(path)：获得绝对路径，等效于增加当前main的路径
    os.path.basename(path)：获得文件名
    os.path.dirname(path)：获得文件路径
    dir, name = os.path.split(path)：获得路径和文件名
    os.path.join(dir,base)：拼接
    os.path.expanduser(path)：替换user为实际路径
    os.path.isdir(path)：路径是否存在
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

关键理解4：(这一点还有一个问题没解决：如何往IDE对应的python3.7中添加PYTHONPATH?)
    (1)main文件在哪里就会以哪个文件夹作为sys.path目录去进行寻找，
    这点可以打印sys.path可以看到每次在哪里运行main，这个路径就会自动被添加到sys.path
    
    (2)基于第一点，其他相对路径就必须要么在该main文件所在文件夹下面，要么在sys.path其他
    文件夹下面，能够拼接成完整路径。此时对于from model import xxx这种语句就容易找不到model
    文件夹的上级路径。必须把源码文件夹根路径作为统一的父路径添加到sys.path中。
    
    (3)还有很重要一点，不同版本的python对应了不同的sys.path和不同的PYTHONPATH。
    比如在命令行输入python是进入默认的python3.5，此时的sys.path，PYTHONPATH是一组。
    而输入python3进入的是python3.7，此时的sys.path和PYTHONPATH是另外一组
    这里需要指出，PYTHONPATH默认是空的，通过添加PYTHONPATH后，会自动导入到sys.path中
    
    (4)基于第2点，添加路径到sys.path的方法有：  
        1. 临时只针对当前终端
        sys.path.insert(0, path), 可以临时看到
        或者在命令行export PYTHONPATH=/home/xx/../xx  # 相当于添加一个临时变量，可通过env指令查看到
        最快的一句：
            sys.path.insert(os.pardir)   # 也就相当于把根目录加入sys.path，其中os.pardir='..', os.curdir='.'
        
        
        2. 永久针对当前用户
        gedit ~/.bashrc                    # 这是打开用户目录~/下的bashrc文件
        export PYTHONPATH=/home/xx/../xx
        
        3. 永久针对所有用户
        sudo gedit /etc/profile            # 这是打开根目录/etc下的profile文件, 而sudo nano /etc/profile往往什么都不显示，可能nano没有gedit好用
        export PYTHONPATH=/home/xx/../xx
        source /etc/profile                 # 立即生效
        
    但要注意：无论用上面那种方法，要搞清楚是添加到哪个python，一定要添加到自己在IDE中使用的那个python中去。
    (我添加总是会添加到python3.5，可我IDE用的是python3.7，所以始终不成功，后来发现原来在我的~/.bashrc文件
    中不知道什么时候我有添加一行alias python='/usr/bin/python3.5'，也就是被我自己指定了python3.5，把这行
    注释掉以后，终端的python也就变成anaconda版本的python3.7了)
    其中在终端输入python进入的是python3.5，且跟anaconda无关。
    而在spyder中查找python版本，则是python3.7,且注明是anaconda自带的。
    
 
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

1. import相关的：符合两入口原则()，且在写相对目录时，因为有隐式相对导入和显式相对导入，所以可以写也可不写父目录地址。
    但绝对不要把入口目录包含进去，比如入口是sys.path，则不要有./..到这一级
    (这点与open的写法正好相反，open的写法是一定要用./..作为开头来代表sys.path的目录才行)
    
    >情况1：导入同一个包里的同级文件，直接用该文件名进行导入即可成功
        比如：from B03_dataset_transform import VOCDataset
        但要注意：
        被导入的文件是不是也包含import语句，他所import的子文件内部是不是有导入语法不合格的问题
        一个文件作为__main__运行时，该文件中不能有相对导入，因为此时相对导入就相当于是引入了上层目录，
        这个上层目录既不是main入口之下的，也不是sys.path入口之下的，不符合两入口原则所以必然报错
        但如果__main__在根目录，这个文件只是被作为模块导入，则不会报错，因为此时该文件的__main__入口权限够高
    >情况2：导入同一个包里的同级文件夹下的文件，直接用该文件夹名进行导入即可成功 (但要确保被导入文件内部没有从当前__main__文件再往上去导入，否则报错)
        比如：from config.config import Config
    >情况3：导入上级文件或其他包里的文件，需要找到共同的父包，然后把这个父包加到sys.path
   然后在该父包之下的所有文件夹/文件都可以直接调用，且不用写父包的名字

2. 文件open相关的：
    同样要满足2个入口理论，唯一区别在于一定要把入口目录包含进去，比如：
    入口在sys.path，则目录一定要把sys.path包含进去，用./..代替(没有隐式打开目录的方法)
    另外注意open所用的相对路径也是想对于__main__文件，而不是相对于调用这个路径的文件(这一点经常搞错)
    因此不同__main__文件可能会导致同一调用文件所用的path也不同
    比如CocoDataset要open数据集地址，如果是根目录的train作为__main__运行，则path='./data/coco'
    而如果是在子目录test/test.py作为__main__运行，则path='../data/coco'

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


# %%
'''
Q: 如何保存变量和加载变量
1. pkl格式的优点：可以存储python认可的所有对象，包括list/dict/object/set等等
   如果是其他如tensor/ndarray, 可以放在list容器里边再存就可以
2. 格式简单便于操作

pickle.dump(var_name, f_handle)
var = pickle.load(f_handle)
其中f_handle = open('file_path', 'type'), type可以是r/rb/rt/r+, w/wb/wt/w+, a/ab/at/a+
'''
import pickle
bboxes = [[263, 211, 324, 339], [165, 264, 253, 372], [241, 194, 295, 299]]
# 最简单最快的方式：
# 保存变量
pickle.dump(bboxes, open('bboxes.txt','wb'))
# 加载变量
bb1 = pickle.load(open('bboxes.txt','rb'))

# 相对正规的方式
with open('./bboxes.pkl','wb') as f:
    pickle.dump(bboxes, f)

with open('./bboxes.pkl','rb') as f:
    pickle.load(bboxes, f)

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
"""如何保存加载yaml文件
1. yaml格式的文件，保存方式采用冒号和缩进的方式表示层级关系，其中冒号前面表示dict的key，冒号后边表示dict的value
冒号后边可以直接写value但必须空一个空格再写value，或者写到下一行用缩进来表示，一个缩进在yaml lib是4个空格，在其他
库有可能是2个空格，此时上一行的冒号后边就不需要跟一个空格了。具体缩进多少其实都可以，只要每层的空格对齐了就行。
例如如下一个yaml格式文件，就是一个典型.yml文件的内容：

model:
    arch: fcn8s
data:
    dataset: pascal
    train_split: train_aug
    val_split: val
    img_rows: 'same'
    img_cols: 'same'
    path: /home/ubuntu/suliang_git/pytorch-semseg/data/VOCdevkit/VOC2012/
    sbd_path: /private/home/meetshah/datasets/VOC/benchmark_RELEASE/
training:
    train_iters: 300000
    batch_size: 1
    val_interval: 1000
    n_workers: 16
    print_interval: 50
    optimizer:
        name: 'sgd'
        lr: 1.0e-10
        weight_decay: 0.0005
        momentum: 0.99
        
2. 采用yaml库对yaml文件进行操作: 读取的数据是一个嵌套的dict
    数组：采用 - 开始
    字符串：默认不带引号(带上也不会出错)，但如果是字符串中有空格或特殊字符(\n)则需要带引号，
    一般是用单引号，虽然双引号也可以但双引号不能对特殊字符转义
例如如下数据格式：

string:
    'eason'
string2:
    'hello world'
array:             # 这是二维list
    -
        - cat
        - dog
        - goldfish
array2:           # 这是一维list
    - 2.0
    - 3.5
    - 1.2
array3: [apple, pearl]

读取方式1：yaml.load()但这种方式虽然能用，但已经被yaml库废弃掉，因为不安全的问题，参考https://msg.pyyaml.org/load
with open('data.yml') as f:
    cfg = yaml.load(f)
读取方式2： yaml.save_load()这是替代方式

3. yml文件结合Dict来用，非常方便，很容易就实现.py文件的cfg导入的功能。
   因为yaml格式专门用来做配置文件的格式，所以还有很多处理格式的小语法，非常方便
    
"""
import yaml
from addict import Dict

with open('data.yml') as f:
    cfg = yaml.load(f)
    cfg = Dict(cfg)



                             
                             
                             