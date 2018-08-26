#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:30:45 2018

@author: suliang
"""

# decorator

'''
Q: 如何在不修改目标函数的基础上，监控目标函数？
'''
# 早期处理方法：新定义了一个debug监控函数来调用目标函数，并在监控函数内不增加需要的手段
def debug(func):
    def wrapper():
        print("[DEBUG]: enter {}()".format(func.__name__))
        return func()
    return wrapper

def say_hello():
    print("hello!")

debug(say_hello)

# 最新的处理方法：新定义一个debug监控函数，但通过@debug语法糖，把监控函数粘到原始函数上
# 相当于给原始函数增加一层外皮，调用原始函数就相当于同时也调用了监控函数
# 装饰器作用：1. 不更改原始函数基础上，就能增加功能；2.批量更改各种函数增加相同功能(各种函数只增加一个@func就行)
def debug(func):
    def wrapper(*args, **kwargs):
        print("[DEBUG]: enter {}()".format(func.__name__))
        return func()
    return wrapper

@debug        # 语法糖：把装饰函数跟原始函数粘在一起
def say_hello():
    print("hello!")
    
say_hello()  # 可以看到调用原函数就有了额外的装饰器函数的功能了


'''
Q: 如何？
'''