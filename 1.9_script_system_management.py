#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:24:38 2018

@author: ubuntu
"""

'''
Q: 如何通过命令行运行python程序？
'''
# 基础运行方式： python3 + xxx.py
python3 main.py

# 带参运行方式： python3 -m + xxx.py
# -m表示以脚本的方式运行模块，所以只需要输入模块名，系统自动添加.py
# 同时-m方式会额外导入python运行环境的当前路径(此处不太明白，等以后再说)
python3 -m visdom.server

# 不中断运行方式：nohup python + xxx.ppy，其中nohup代表不挂起，terminal关闭也会运行
# & 代表后台运行
nohup python -m visdom.server &


'''
Q: 如何通过命令行输入参数给程序？
    输入python3 demo.py  通常可以显示需要输入哪些参数
    
    -argu 代表可选参数的短参数形式，默认可以不输入。输入形式：-argu 2
    --argu 代表可选参数的长参数形式
    argu 代表位置参数，默认为必须输入。输入形式为：xxx, 即直接输入
    
    每种参数的可选属性：
    type=float 代表参数类型
    default='voc' 代表参数默认值 (但似乎只有可选参数能定义default???)
    action="store_true" 代表可以不输入参数
    nargs = '?' 代表输入参数个数
    choices=[0, 1, 2] 代表参数输入可选范围
    help = 'input a string'
'''
import argparse                               # 第一步：导入库
parser = argparse.ArgumentParser()            # 第二步：创建参数容器
parser.add_argument('--gpu', type=int, default=-1)  # 第三步：添加参数
parser.add_argument('--pretrained-model', default='voc07')
parser.add_argument('image', nargs='?', default='example.jpg')
args = parser.parse_args()                    # 第四步：创建参数对象


'''
Q. 如何创建symlink软连接？
'''
$cd slcv
$mkdir data
$ln -s /home/ubuntu/MyDatasets/coco
$ln -s /home/ubuntu/MyDatasets/voc/VOCdevkit



'''
Q. 如何在代码中运行命令行指令？
是否可以使用
'''




'''
Q. 如何设置ipython中显示的图片是单独显示而不是嵌入在命令行？
参考：https://blog.csdn.net/mozai147/article/details/79850065
'''
#命令行输入：代表嵌入式显示
%matplotlib inline
#命令行输入：代表独立显示
%matplotlib qt5
# spyder中设置：inline代表嵌入显示，qt5代表独立显示
Tools > Preferences > IPython Console > Graphics > Graphics backend
