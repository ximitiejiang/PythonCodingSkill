#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:30:45 2018

@author: suliang

"""

# decorator

'''
Q: 如何在不修改目标函数的基础上，监控目标函数？

核心概念：使用装饰器，在不改变原函数情况下，给原函数增加功能
装饰器就是一个新函数，它接受一个原函数名作为输入，并返回一个新函数作为输出。参考python cookbook9.1

核心理解1：书写逻辑是先做个装饰函数@new_func，再在new_func中定义wrapper()函数并返回wrapper函数名
           最后考虑wrapper()函数写法做2件事(新功能+原函数返回)
核心理解2： 参数传入的是原函数名，参数传出的也是wrapper函数名，而不是函数调用
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
# 装饰器作用：1. 不更改原始函数基础上，就能增加功能；
#           2.批量更改各种函数增加相同功能(各种函数只增加一个@func就行)
def debug(func):
    def wrapper(*args, **kwargs):  # 定义外皮
        print("[DEBUG]: enter {}()".format(func.__name__))  # 
        return func(*args, **kwargs)
    return wrapper
@debug        # 语法糖：把装饰函数跟原始函数粘在一起
def say_hello():
    print("hello!")
    
say_hello()

# 另一个实例: 可以用它作为装饰器模板，所有名称都不要改，只需要改print这部分执行代码。
def addthing(func):
    def wrapper(*args, **kwargs):
        print('add something here!')
        return func(*args, **kwargs)
    return wrapper

@addthing
def say_age(father=0.0, mother=0.0, son=0.0):
    print(father, mother, son)

say_age(30, 28, 2)


'''
Q. python自带哪些可用的系统装饰器和装饰器函数？

- @property: 属性化的普通方法(self隐含)，相当与定义了property()作为装饰器函数，里边直接调用getter()函数，所以可不带括号调用
- @staticmethod: 静态方法(无隐含)，相当与定义了staticmethod()作为装饰器函数，该函数新功能就是对输入做判断，不直接接受对象的属性
- @classmethod：类方法(cls隐含)，相当与定义了classmethod()作为装饰器函数，该函数新功能就是对输入做判断，不直接接受对象的属性

'''
class Person():
    @property
    def name(self):
        print('Leo')
p = Person()
p.name


"""
Q. 如何用装饰器作为在调试时计时器？
"""
import time
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func()
        print('this function lasts time: {}'.format(time.time()-start))
    return wrapper

@timeit
def func():
    time.sleep(2.0)

func()


"""
Q. 如何对类的方法添加装饰器？
"""
def timeit(func):
    def wrapper(me_instance):  # wrapper()函数传入的参数，需要是一个对象，要用me_instance代替
        start = time.time()
        func(me_instance)      # 类方法的调用，也需要加入me_instance来代表对象参数self
        print('this function lasts time: {}'.format(time.time()-start))
    return wrapper

class Sleep:
    @timeit
    def sleep(self):
        time.sleep(2.0)
s = Sleep()
s.sleep()
