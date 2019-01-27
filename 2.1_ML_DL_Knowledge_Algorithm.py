#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 09:43:06 2019

@author: ubuntu

这部分主要用来从基本概念角度，通过实际代码实践，深刻理解代码
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
1. 卷积操作：用来对图像进行过滤，
"""
# 尝试最简单的几种卷积核：可以看到卷积核能够对图像进行过滤



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
"""



# %%        网络训练
"""样本不平衡问题是什么，影响是什么，怎么解决样本不平衡问题？
"""



