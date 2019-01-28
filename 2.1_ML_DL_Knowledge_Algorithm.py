#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 09:43:06 2019

@author: ubuntu

这部分主要用来从基本概念角度，通过实际代码实践，深刻理解代码
    1. 概率论
    2. 传统机器学习
    3. 网络基础层
    4. 损失函数
    5. 正则化
    6. 激活函数
    7. 反向传播与梯度下降
    8. 网络训练
"""

# %%        概率论
"""什么是随机变量？有哪几种随机变量，有什么用？
1. 随机变量是
2. 两种随机变量
    > 离散型随机变量
    > 连续型随机变量
"""
# 待增加代码

# %%        传统机器学习算法
"""如何把高维变量映射到低维？以及什么是PCA？
"""

# %%        网络基础层
"""卷积的作用?
参考：https://blog.csdn.net/zouxy09/article/details/49080029
1. 卷积操作作用：卷积在cv领域也叫模板，是已知卷积参数对图像进行过滤计算，得到指定特征。说明了卷积参数能够代表对应特征
   而在神经网络领域是已知特征，通过卷积操作后的损失函数进行梯度下降，得到卷积参数，这个参数就代表了卷积核学到的特征
   由于存在各种各样的卷积核，可以学习到图像的边缘/颜色/形状/纹理等各种模式，并通过组合各种模式会得到更高语义的特征，
   而神经网络会对每一个通道(开始的3层，到64层，到128层，层数越来越多，目的就是增加越来越多的卷积核)分别卷积
2. 卷积核的特点：
   卷积核所有元素之和一般等于1，这样能保证卷积前后图片亮度相同，如果大于1则图片变亮，反之变暗
   卷积后可能会出现负数或者大于255的数，这时需要截断操作到0-255
3. 特定的卷积核能够对图形进行过滤，得到比如水平边缘线/竖直边缘线/整个边缘线等
4. 水平滤波之所以能够找到水平线，是因为在竖直方向上原像素位置取正其他取-1，相当离散版的求导：分别取像素竖直方向前后2点的差，相当与斜率。
   垂直滤波的逻辑也是一样的，只要保证和为0(黑化非边沿像素)，求导方向与线方向垂直(垂直于线方向就是像素变化最大方向，也就是梯度方向)
5. 平均值滤波一方面可以模糊化图片，同时进行局部平均值滤波有去除噪声的效果：因为噪声被认为是零平均值的随机变量，局部平均后噪声就被置0了
6. opencv自带卷积操作函数des=cv2.filter2D(src, -1, kernel)，但该函数只支持单通道所以需要先img的bgr分离或者用灰度图
7. 神经网络的卷积操作，就是用来卷积参数来学习图像特征。
    > conv = nn.Conv2d(in_c, out_c, k_size, stride=1, padding=0, dilation=1, bias=True)，这是默认参数的值
    > 输出层数：in_c自由定义
    > 输出尺寸：out_c计算得到，Hout = (Hin - k_size + 2p -1)/s + 1
    > 卷积参数：默认是s=1,p=0但这样会导致图像尺寸缩小，所以通常用s=1,p=1这样能保证图像尺寸不变
    > dilation/bias当前一般用默认参数, 也就相当那个与d=1,b=1（dilation负责卷积的时候扩大范围，bias负责是否增加偏置参数）
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('test/test_data/sudo.jpg', 0)
# 常见卷积核
conv_median = (1/9)*np.array([[ 1, 1, 1],   #这个是中值滤波：亮度相同, 取平均相当与像素点平均化，产生模糊化效果
                              [ 1, 1, 1],   # 
                              [ 1, 1, 1]])
conv_sharp = np.array([[-1,-1,-1],          #这个是锐化滤波：锐化类似边缘检测，把水平垂直倾斜的线条都识别出来，然后+1，则图片亮度不变但会更锐化
                       [-1, 10,-1],          # 卷积核越大效果越明显
                       [-1,-1,-1]])
conv_edge_all = np.array([[0,-4, 0],          #这个是整体边缘检测，类似锐化滤波，差别是总和设为0，则图片变暗突出线条
                          [-4,16,-4],          # 卷积核越大效果越明显
                          [0,-4, 0]])
conv_edge_x = np.array([[0,0,0,0,0],          #这个是水平边沿滤波：变暗，突出边沿线
                      [0,0,0,0,0],            
                      [-1,-1,4,-1,-1],
                      [0,0,0,0,0],
                      [0,0,0,0,0]])
conv_edge_y = np.array([[0,0,-1,0,0],          #这个是水平边沿滤波：变暗，突出边沿线
                       [0,0,-1,0,0],
                       [0,0, 4,0,0],
                       [0,0,-1,0,0],
                       [0,0,-1,0,0]])
conv_moving = (1/5)*np.array([[1,0,0,0,0],          #这个是运动模糊滤波：沿着45方向取同一直线的像素做平均，效果类似图像沿45度移动
                             [0,1,0,0,0],           # 卷积核越大，影响的像素越多，效果越明显
                             [0,0,1,0,0],
                             [0,0,0,1,0],
                             [0,0,0,0,1]])
conv_big_moving = (1/10)*np.eye(10)

# 卷积核的基本运算逻辑：对应位置相乘后累加
data1 = np.eye(5,5)
conv = np.array([[-1,-1,-1],          
                 [-1, 10,-1],          
                 [-1,-1,-1]])
res1 = cv2.filter2D(data1, -1, conv_sharp)  # 在filter2D()函数中默认是添加padding(0)保证输出与输入的尺寸一样
                                            # 手算第一个数：10*1+(-1)=9 但函数算出来第一个数是6，不过中间的数算出来是对的。
                                            # 似乎filter2D函数的padding策略不同，导致边沿的数算出来不太一样。
# 卷积核用在真实图片的效果：可以看到卷积核能够对图像进行过滤
res2 = cv2.filter2D(img, -1, conv_edge_x)
res3 = cv2.filter2D(img, -1, conv_edge_y)
res4 = cv2.filter2D(img, -1, conv_median)
res5 = cv2.filter2D(img, -1, conv_moving)
res6 = cv2.filter2D(img, -1, conv_big_moving)
res7 = cv2.filter2D(img, -1, conv_sharp)
res8 = cv2.filter2D(img, -1, conv_edge_all)

plt.subplot(421), plt.imshow(img, cmap='gray'),plt.title('original')
plt.subplot(422), plt.imshow(res2, cmap='gray'),plt.title('conv x')
plt.subplot(423), plt.imshow(res3, cmap='gray'),plt.title('conv y')
plt.subplot(424), plt.imshow(res4, cmap='gray'),plt.title('median')
plt.subplot(425), plt.imshow(res5, cmap='gray'),plt.title('moving')
plt.subplot(426), plt.imshow(res6, cmap='gray'),plt.title('big moving')
plt.subplot(427), plt.imshow(res7, cmap='gray'),plt.title('sharp')
plt.subplot(428), plt.imshow(res8, cmap='gray'),plt.title('edge_all')

# pytorch的卷积核操作基本参数：
import torch
import torch.nn as nn
from numpy import random
# 用conv默认参数：s=1,p=0,d=1,b=True，默认参数的问题是无法保持图形尺寸不变
input = torch.tensor(random.uniform(-1,1,size=(8,16,50,100)).astype(np.float32))    # input = (8,16,50,100) b,c,h,w
conv = nn.Conv2d(16, 33, 3, stride=1, padding=0, dilation=1, bias=True)             
output = conv(input)                                                                # output = (8,33,48,98) b,c,h,w， 其中h=(50-3)/1 + 1=48
# 修改默认参数：s=1,p=1，这样能够保证输出图形尺寸不变
input = torch.tensor(random.uniform(-1,1,size=(8,16,50,100)).astype(np.float32))    # input = (8,16,50,100) b,c,h,w
conv = nn.Conv2d(16, 33, 3, 1, 1)                                                   
output = conv(input)                                                                # output = (8,33,50,100) b,c,h,w， 其中h=(50-3+2)/1 + 1=50



# %%        网络基础层
"""下采样和上采样的作用？通常如何实现？
1. 下采样：是指缩小图像，也叫降采样(downsample/subsample)，主要目的是使图像尺寸缩小，
    > 下采样之前一般用nn.MaxPooling(k_size, s=None, p=0, d=1, ceil_mode=False)来定义s=2来实现
                 或用nn.
                 下采样池化的Hout = (Hin - k_size + 2p -1)/s + 1
    > 下采样现在一般用nn.Conv2d()定义s=2来实现
2. 上采样：是指放大图像，也叫图像插值(umsample/interpolating)
    > 上采样在
    > 上采样还可用bilinear
3. 下采样池化层，可以有maxpooling(), averagepooling()
    > 池化层不含可学习参数，主要目的是仿照人类视觉系统，对输入进行降采样，做法就是
      取局部最大值，或者局部平均值，这样做好处包括：
      特征不变性(用最大值/平均值来代表特征，而不是具体位置的数据，一定程度使学习有一定的空间自由度)，
      特征降维(下采样后尺寸减小，参数个数减少)，
      一定程度防止过拟合(...)
"""
import torch
import torch.nn as nn
from numpy import random
# 最大池化和平均池化

random.seed(3)
data = random.uniform(1,50, size=(3,5,5)).astype(np.float32)
input = torch.tensor(data)  # (c,h,w)=3,5,5
mpool = nn.MaxPool2d(2)
apool = nn.avgPool2d(2)
output1 = mpool(input)   # 最大池化
output2 = apool(input)   # 平均池化


# %%        网络基础层
"""BatchNorm2d
"""


# %%        网络基础层
"""为什么当前所有神经网络都是设计成很多基础层叠加，形成很深的神经网络，并且特征尺寸越来越小，特征通道越来越多？
1. 多层神经网络的优点，一个核心就是可以有扩展的感受野
2. 感受野：是指后一层一个节点在前一层或前n层所对应的特征尺寸大小，就是这个节点在前n层的感受野大小
   比如一个7x7卷积核的后一层单个节点对应前一层感受野大小就是7x7，而如果是3x3卷积核往后3层的节点也能在前3层得到同样7x7的感受野(节点-3x3-5x5-7x7)
3. 虽然小卷积核与大卷积核都能得到相同感受野，但小卷积核有很多优势：
    > 小卷积核需要多层叠加形成大的感受野，多层能够
"""


# %%        网络基础层
"""
"""



# %%        激活函数
"""解释不同激活函数的优缺点？
1. 激活函数的功能：也叫非线性映射函数，
1. 常见激活函数
    > sigmoid()
    > relu()
    >
2. 
"""
# 如果没有激活函数，无论多少层都只是线性变换




# %%        损失函数
"""熵的概念和计算，以及交叉熵概念，以及交叉熵作为损失函数的意义？
1. 信息熵概念：是指某一个离散型随机变量X的不确定性的度量，随机变量X的概率分布为p(X)=pk (k=1..n) 
    >Ent(X)= - sum(pk*log(pk)), 其中Ent(X)就是系统的信息熵，pk代表每一个事件的概率
     从公式可看出信息熵支持多分类标签的计算(而基尼指数公式就只支持二分类)
    >每一个数据集看成一个系统就有一个信息熵，每一个子集也有一个信息熵，都代表了这个系统基于这个分类方式的混乱程度
    >
2. 信息增益：是指系统基于某种特征划分下的信息熵增益
    >gain = Ent(D) - sum((Di/D)*Ent(Di)), 其中Ent(D)为集合的信息熵，而(Di/D)*Ent(Di)为某一子集合的条件熵
     可理解为集合在加入某特征后不确定度减小的程度，也就是增益。

3. 交叉熵：是
    >
4. 相对熵：也叫KL散度
"""
def calEntropy(data, n):
    """计算一个数据集作为随机变量的信息熵, data为mxn array, n为第n列作为随机变量的概率事件"""
    from math import log
    numEntries = len(data)
    labelcounts = {}  # 字典用于放置{总计多少类：每个类多少样本个数}
    for line in data: # 循环取每一行样本
        currentlabel = line[-1]
        if currentlabel not in labelcounts.keys(): # 如果是一个新分类
            labelcounts[currentlabel] = 0       # 在字典新增这个新分类，对应样本个数0
        labelcounts[currentlabel] +=1    # 如果不是新分类： 在旧类的样本个数+1 
    # 计算一个数据集的信息熵
    shannonEnt = 0.0
    for key in labelcounts:
        pi = float(labelcounts[key])/numEntries # 计算每一个分类的概率p=分类数/总数
        shannonEnt -= pi * log(pi,2)    # 计算entropy = -sum(p*logp)
    return shannonEnt

def calCrossEntropy():
    pass

def test():
    # 读取数据
    persons = []
    cols = ['age', 'has_job', 'has_house','has_loan', 'approved']  # 用随机变量来理解数据：随机变量就是贷款能否获得批准的事件的概率
    with open('test/test_data/loan.txt') as f:
        loans = f.readlines()
        for loan in loans:
            loan = loan.split()
            persons.append(loan)
    e0 = calEntropy(persons)
    print('total entropy: {:.6f}'.format(e0))


# %%        损失函数
"""损失函数的定义是什么？为什么要有损失函数？
1. 损失函数：是指所有样本的预测值与标签值之间的偏差函数。这种偏差通常采用不同的方式评估
   比如通过欧式距离最小来评估，比如通过交叉熵最小来评估
   损失函数是关于一组参数(w1..wn)的函数，为了预测最精确就是让损失函数最小，所以通过求解损失函数的最小值来得到这组参数。
2. 逻辑回归的过程：一组数据(x1,..xn)经过线性映射到h(w) =w1x1+w2x2..+wnxn, 再经过非线性映射g(theta)=sigmoid(h)
   这样就建立(x1..xn) -> h(w1..wn) --> g(theta) 的映射，我们认为g就是预测值，损失函数采用欧式距离评估
   因此通过找到损失函数最小值，来得到最小值时的参数(w1..wn)
3. 如何选择损失函数：
    >
"""
# 损失函数之1：欧式距离MSE


# 损失函数之2：交叉熵


# %%        正则化
"""解释l0/l1/l2正则化，以及如何在深度学习网络中使用？
"""





# %%        反向传播与梯度下降
"""凸函数概念，求极值方法，为什么用梯度下降而不是牛顿法？
"""


# %%        反向传播与梯度下降
"""CNN的算法原理, BP相关理论的python如何实现?
"""
class CNN:
    def __init__(self):
        pass







# %%        网络训练
"""过拟合与欠拟合的概念，原因，解决办法？
"""


# %%        网络训练
"""网络参数初始化的方法有哪些，如何选择初始化方法
"""


# %%        网络训练
"""网络超参数有哪些，如何定义超参数？
1. 模型各层参数：
    > Conv2d()
    > MaxPooling2d()
"""



# %%        网络训练
"""样本不平衡问题是什么，影响是什么，怎么解决样本不平衡问题？
"""



