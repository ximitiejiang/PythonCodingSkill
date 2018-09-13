#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:29:58 2018

@author: suliang
"""

'''
Q: 如何绘制几种最常见曲线？
'''
import matplotlib.pyplot as plt

def regressionData():
    from sklearn.datasets import make_regression
    X,y,coef=make_regression(n_samples=1000,n_features=1,noise=10,coef=True)
    #关键参数有n_samples（生成样本数），n_features（样本特征数）
    # noise（样本随机噪音）和coef（是否返回回归系数)
    # X为样本特征，y为样本输出， coef为回归系数，共1000个样本，每个样本1个特征
    return X, y

# 连线图
X, y = regressionData()
plt.plot(X,y, )

# 