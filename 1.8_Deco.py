#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:30:45 2018

@author: suliang

"""

# decorator

'''
Q: 如何在不修改目标函数的基础上，监控目标函数？

核心概念：使用装饰器，在不改变原函数情况下，给原函数进行wrap(翻译成封装)，原函数是wrapped，返回的新函数叫wrapper
装饰器就是一个新函数，它接受一个原函数名作为输入，并返回一个新函数作为输出。参考python cookbook9.1
核心理解1： 参数传入的是原函数名，参数传出的也是wrapper函数名，而不是函数调用

核心理解1：书写逻辑是先做个装饰函数@new_func，再在new_func中定义wrapper()函数并返回wrapper函数名
           最后考虑wrapper()函数写法做2件事(新功能+原函数返回)
核心理解2： 参数传入的是原函数名，参数传出的也是wrapper函数名，而不是函数调用

核心理解3：装饰器任务是高阶函数，对函数进行二次加工，所以整个设计过程是：
            1. 约定俗成的命令规范：外层函数自定义一个作为装饰器功能名称，内层函数叫wrapper函数，用来替代原函数(原函数也叫wrapped)
            2. 先写一对嵌套def分别得到原函数f和原函数参数。
            3. 然后操作函数f(*args,**kwargs)的返回值，可以直接返回，可以加工返回，可以不返回，可以加别的额外语句
            4. 最后返回wrapper
            
def named_deco(wrapped):        # named_deco是装饰器功能函数名， wrapped是原函数名(被封装的)
    def wrapper(*args, **kwargs):  # wrapper是内部封装函数
        return wrapped(*args, **kwargs)
    return wrapper                 # 最终返回封装函数

对应的functools的高阶函数wrap(wrapped), update_wrapper(wrapped, wrapper)也都是操作wrapped或wrapper这两个函数对象

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


'''-------------------------------------------------------------------------
Q. python自带哪些可用的系统装饰器和装饰器函数？
(这部分详细说明在 7_Class_and_Object)
@property:      属性化的普通方法(self隐含)，相当与定义了property()作为装饰器
                函数，里边直接调用getter()函数，所以可不带括号调用
@staticmethod:  静态方法(无隐含)，相当与定义了staticmethod()作为装饰器函数，
                该函数新功能就是对输入做判断，不直接接受对象的属性
@classmethod：  类方法(cls隐含)，相当与定义了classmethod()作为装饰器函数，该
                函数新功能就是对输入做判断，不直接接受对象的属性
关键3：区分3大方法   
     * 类方法，classmethod，必写的隐含参数是cls，不可以访问对象属性，可以被类和对象调用
     * 普通方法(也叫实例方法)，必写的隐含参数是self，可以访问任何属性，只可以被对象调用
         * 属性方法属于普通方法的一种，所以必写隐含参数self
     * 静态方法，staticmethod，没有必写隐含参数，不可以访问类或对象的任何属性/方法，可以被类和对象调用
       但可以传入对象self, 然后再调用对象属性。
       也可传入类cls，然后调用类属性
                
     * @property属性方法：是普通方法的一种，主要用来把方法属性化，调用方便省去写括号，a.action
     * @classmethod类方法：用来获得类的一些基本属性展示
     * @staticmethod静态方法：用来剥离出来放一些独立逻辑，跟类和对象都不会交互，但可以被大家使用。
        
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
        func(*args, **kwargs)
        print('this function lasts time: {}'.format(time.time()-start))
    return wrapper

@timeit
def func():
    time.sleep(2.0)

func()


'''------------------------------------------------------------------------
Q. 多层装饰器嵌套时的执行顺序是怎么样的？
'''
def func1(src):
    print('func1')
    def wrapper(*args, **kwargs):
        print('wrapper1') 
        src(*args, **kwargs)
    return wrapper

def func2(src):
    print('func2')
    def wrapper(*args, **kwargs):
        print('wrapper2') 
        src(*args, **kwargs)
    return wrapper

        
@func2
@func1   # 先执行上面的wrapper2然后下面的wrapper1
def test():
    print('test')

test()     #最终输出顺序是： func1 -> func2 -> wrapper2 -> wrapper1 -> test


"""-----------------------------------------------------------------------
Q. 如何设计带参装饰器？
"""
def func(*dargs, **dkwargs):
    


@func(1,a=2)
def test():



"""-------------------------------------------------------------------------
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


"""------------------------------------------------------------------------
Q. 如何使用functools里边的update_wrapper()高阶函数？
1. 使用functools.update_wrapper的原因：在原函数增加wrapper以后，
原函数的函数名__name__以及注释__doc__都会变成wrapper装饰器返回的inner函数的信息
2. update_wrapper(wrapper, wrapped)属于高阶函数，用来从原函数的属性如name/doc/dict/module拷贝给wrapper函数，使返回的wrapper函数具备跟原函数相同属性
3. 核心理解：update_wrapper()跟wraps()的区别：update_wrapper()可作为函数独立运行，也无需返回什么
   而wraps()通常只作为@wraps(func)这样的装饰器放在函数定义的前面。
   如果没有函数定义而只有一个函数返回，比如partial()作为wrapper，不能在前面加@wraps()，但可以用update_wrapper()
"""
from functools import update_wrapper
def test(functest):
    """this is test decorator __doc__"""
    def wrapper(*args, **kwargs):
        """this is wrapper __doc__"""
        return functest(*args, **kwargs)
    update_wrapper(wrapper, functest)      # 负责把原函数func的相关属性如name/doc/dict/module拷贝给目标函数
    return wrapper

@test
def functest():
    """this is functest __doc__"""
    pass

print(functest.__doc__)   # 如果不增加update_wrapper则注释输出inner()的注释，增加后输出原来functest()的注释
print(functest.__name__)  # 如果不增加update_wrapper则函数名输出inner函数名，增加后输出原来functest函数名


"""------------------------------------------------------------------------
Q. 如何使用functools里边的wraps()
1. 核心理解：等效于用update_wrapper()函数对wrapper进行了二次wrap, 效果跟update_wrapper()一样
2. 核心理解：update_wrapper()跟wraps()的区别：update_wrapper()可作为函数独立运行，也无需返回什么
   而wraps()通常只作为@wraps(func)这样的装饰器放在函数定义的前面。
   如果没有函数定义而只有一个函数返回，比如partial()作为wrapper，不能在前面加@wraps()，但可以用update_wrapper()
"""
from functools import wraps
def wrapper(functest):
    """this is wrapper __doc__"""
    @wraps(functest)      # 负责把源函数的信息封装给装饰器待返回的内函数，等效于update_wrapper()函数
    def inner(*args, **kwargs):
        """this is inner __doc__"""
        return functest(*args, **kwargs)
    return inner

@wrapper
def functest():
    """this is functest __doc__"""
    pass

print(functest.__doc__)   # 如果不增加update_wrapper则注释输出inner()的注释，增加后输出原来functest()的注释
print(functest.__name__)  # 如果不增加update_wrapper则函数名输出inner函数名，增加后输出原来functest函数名
