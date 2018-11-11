#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:29:58 2018

@author: suliang

汇总主要函数
* list
* dict
* string
* array
* dataframe
* python基础函数

"""


'''##########################################
Q: Python的基本函数？
'''
import numpy as np
a = [1,4,-3,2,7,-5,9,2]

# numpy中以下3个不用括号
b = np.array(a)
b.shape
b.dtype
b.T

# numpy中以下三个后置式替代python，求和/最大最小/平均值函数，后置式更方便，pytorch也是用这个
b = np.array(a)
b.sum()  # 直接求和写法，对象只能是array不能是list
b.max()  # 求最大值
b.mean() # 求平均值

# python - 基本运算
abs(-2)   # 绝对值
pow(10,2) # 幂指数
sum(a)
range(5)

# python - 取整取余取小数
round(3.14)  #
ceil(3.14)
int(3.14)
math.modf(3.14)

# python - 除法取整取余
x = 37
x//4  # 取相除的整数
x%4   # 取相除的余数

# python - 格式转换
float('3')
str(123)
bin(3)
hex(15)
list('abc')
dict(a=1,b=2)

# python - 判断
a = 3
isinstance(a, int)
all()
any()
where()

# python - 返回不重复值
set(a)   # 用于返回不重复值，并且默认排序，实际上是建立一个set数据结构，

# numpy补充 - 计算


'''##########################################
Q: list的基本函数？
'''
lst = [2,1,2,5,2]
# 两种增加的方式
lst.append(9)  # append是在末尾加入一个新元素
lst.extend([1,2])   # extend是加入一串新元素list
# 计数
lst.count(2)  # 统计有多少个2
# 三种弹出的方式
lst.pop(3)     # 弹出最后一个，或者pop(2)弹出指定位置的元素
lst.remove(2)  # 删除值为2的元素，但每次只删除一个
del lst[3]     # 删除第0个位置的元素
# 三种排序的方式
lst.reverse()  # 反转顺序
lst.sort(reverse = True)     # 排序，inplace的方式
lst.sorted()                # 排序，临时性的
# 复制
lst.copy()


'''##########################################
Q: dict的基本函数？
'''

dict = {'a':1, 'b':3, 'c':2, 'd':5}

dict.keys()
dict.values()
dict.items()

dict.get('c', 0)  # 如果不存在返回值可设置，此处设置为0


'''##########################################
Q: str的基本函数？
'''
str = ' d8,apple,8d '
str.strip()   # 切除，默认切除左右两边的空格/回车/tab
str.lstrip()
str.rstrip()

str.split(',')  # 分隔，基于逗号
str.count('d')


