#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:29:54 2018

@author: suliang
"""

# functions


'''
Q: 怎么编写可接受任意数量的位置参数，或者任意数量的关键字参数？
'''
def avg(*args, **kwargs):  # 位置参数 + 关键字参数(kwargs = keyword args)
    return(sum(args) / len(args))
avg(1,2,3,4)    
    # *args代表位置参数，以一个星号开始的位置参数只能作为最后一个位置参数
    # **kwargs代表关键字参数，以两个星号开始的关键字参数只能作为最后一个参数
    # 强制定义了*args后边，只能是关键字参数，也就只能输入成 name = xxx

def recv(size, *, block):  # 使用星号作为强制的关键字参数的开始
    pass
recv(1024, True)         # 这种写法报错，因为block是关键字参数而不是位置参数
recv(1024, block = True) # 必须是关键字参数写法
    

'''
Q: 函数返回的多个变量是怎么存在的，怎么获得？
'''
def myfun():
    return 1, 2, 3
all = myfun()    # 函数的多个返回值实际上是返回一个元组
x,y,z = myfun()  # 用多个变量接收函数的多个返回值，实际上是元组的解包


'''
Q: 对于一些简单的功能，如果不想编正经函数，有没有简洁方式，比如匿名函数？
'''
add = lambda x, y: x + y  # 可以理解为lambda就是一个匿名函数名，后边跟变量名，分号后跟函数体
add(2,3)


'''
Q: 对于？
'''


