#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:29:54 2018

@author: suliang
"""

# functions


'''
Q: 怎么编写可接受任意数量的位置参数，或者任意数量的关键字参数？
1. 核心概念：
    (1)位置参数def foo(x,y)
    (2)关键字参数def foo(x=1)
    (3)可变位置参数def foo(*args)
    (4)可变关键字参数def foo(**kwargs)
2. 本质
    (1)*args：其中args代表元组，*代表拆包操作，*args代表拆包完成的多个位置参数
        例如：args = (1,2,3)，则*args -> 1,2,3
    (2)**kwargs：其中kwargs代表字典，**代表拆字典操作，**kwargs代表拆字典完成的多个关键字参数
        例如：kwargs={'a':1,'b':2}, 则**kwargs -> a=1, b=2
3. 参数顺序：
    (1)位置参数 -> *args -> **kwargs
    (2)位置参数 -> 关键字参数 -> *args -> **kwargs

4. 核心应用：
    (1)在形参位置，用*args/**kwargs代表可以输入多个位置参数或关键字参数
    (2)在实参位置，用*/**来解包元组或解包字典
    (3)用来在继承类中给父类传递参数，super().__init__(*args, **kwargs)

'''
def avg(*args, **kwargs):  # 位置参数 + 关键字参数(kwargs = keyword args)
    print(args)
    print(kwargs)
avg(1,2,3,4)   
avg(*(1,2), **{'a':3,'b':4})
    # *args代表位置参数，以一个星号开始的位置参数只能作为最后一个位置参数
    # **kwargs代表关键字参数，以两个星号开始的关键字参数只能作为最后一个参数
    # 强制定义了*args后边，只能是关键字参数，也就只能输入成 name = xxx

def recv(size, *, block):  # 使用星号作为强制的关键字参数的开始
    pass
recv(1024, True)         # 这种写法报错，因为block是关键字参数而不是位置参数
recv(1024, block = True) # 必须是关键字参数写法


"""
Q. 如何理解*args和**kwargs的真正应用区别？
关键理解：
*args可以输入一批数据value，并会被自动转化成列表
**kwargs可以输入一批数据key=value，并会被自动转化成字典
"""
# args的功能是自动把输入参数变成了一个可迭代的list
def args_test(param1,*args):
    print("first param is:",param1)
    index = 1
    for value in args:
       print("the "+str(index)+" is:"+str(value))
       index += 1
# kwargs的功能是自动把输入参数变成了一个可迭代的dict
def kwargs_test(param1,**kwargs):
    print("the first param is: ",param1)
    for key in kwargs:
        print("the key is: %s, and the value is: %s" %(key,kwargs[key]))

    

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
add = lambda x, y: x + y  # 可以理解为lambda就是一个匿名函数名，后边跟变量名，冒号后跟函数体
add(2,3)


'''
Q: 对于函数体内的变量的作用域是如何？如何修改函数体外的变量？
'''
def f1():
    value = [100, 100]

def f3():
    global value         # 需要声明一个全局变量出来，才能对这个已有全局变量进行修改
    value = [100, 100]

value = [0,0]
f1()
print('after f1, value = {}'.format(value))

value = [0,0]
f3()
print('after f3, value = {}'.format(value))
