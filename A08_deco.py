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
核心理解0: 
    >装饰器的传递物: 可以是函数(常规装饰器)，用于对该传入函数添加额外功能
     也可以是类(类注册装饰器)，用于汇总所有类名
    >装饰器的接收方: 接收方要做2件事，接收传递物(函数/类)，执行传递物(执行原函数)
     可以是函数(常规装饰器)，该接收方函数可以自定义相关
     也可以是类的方法函数，该接收函数本质跟常规的装饰器接收函数一样，但可以通过类的对象化可以对接收的函数进行分类(一个对象对应一个分类)
     也可以是类(类装饰器)，相当于用类的初始化函数做接收，用类的__call__函数做执行

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


'''----------------------------------------------------------------------
Q. 装饰器实际的执行过程？
0. 核心：
    >装饰器是放置在函数定义处，所以接收的是一个函数名func，跟实参无关，返回的也是一个函数名wrapper
    >外层的return wrapper必不可少，否则函数消失了；内层的return可以没有，取决原始函数是否需要返回值 (***)
    
1. 包装后test=additional_test(test), test(a,b)=additional_test(test)(a,b)
2. 执行test()的实际过程：首先执行additional_test第一句print, 然后中间wrapper函数定义自动跳过
   然后执行第二句print, 然后返回wrapper函数给到test(2,8)这行,也就变成执行wrapper(2,8),
   然后执行wrapper中的print,然后执行最后的test(2,8)
'''
def additional_test(func):
    
    print('start wrap')
    def wrapper(*args, **kwargs):
        print('additional test result is {}'.format(sum(args)))
        func(*args, **kwargs)
        
    print('start additional test')
    return wrapper
    
@additional_test
def test(a,b):
    print('test result is {}'.format(a**2 + b**2))
    return a**2+b**2

test(2,8)


'''------------------------------------------------------------------------
Q. 多层装饰器嵌套时的执行顺序是怎么样的？
1. 多个装饰器可以用函数嵌套理解：test = func1(func2(test))，先把内层的返回函数获得，然后再外层，然后再从外层依次解析wrapper；
想当于一个先内后外(去两层func皮)，再外后内(执行两个wrapper)的过程
所以执行过程是：
    先func2(test)运行，打印pos1, 然后跳过函数定义，打印pos2
    然后返回func2的wrapper到func1
    然后执行func1, 打印pos3, 跳过函数提定义，打印pos4
    然后返回func1的wrapper到主代码
    执行func1的wraper()打印pos5，然后执行wrapper内的函数，跳转到func2的wrapper内
    执行pos 6, 然后执行func1 wrapper内的test()执行pos 7
'''
def func1(src):
    print('func1 started')   # pos 3
    def wrapper(*args, **kwargs):
        print('wrapper1')    # pos 5
        return src(*args, **kwargs)
        
    print('func1 finished')  # pos 4
    return wrapper

def func2(src):
    print('func2 started')  # pos 1
    def wrapper(*args, **kwargs):
        print('wrapper2')   # pos 6
        return src(*args, **kwargs)
        
    print('func2 finished') # pos 2
    return wrapper
  
@func1
@func2
def test():
    print('test')   # pos 7

test()     #最终输出顺序是： pos 1,2,3,4,5,6



'''------------------------------------------------------------------------
Q. 更普遍而本质的方式理解装饰器：就是传入一个对象(可以是一个类/一个函数)？
1. 装饰器本身：本质是一个可以接收参数的函数，所以可以是函数，也可以是带__call__的类，也可以是对象的方法
2. 装饰器在@deco这一步运行时，就已经运行，并准备好了相关返回的wrapper
2. 传入装饰器的参数：可以是一个函数，从而组成嵌套函数；也可以是类名(cls)；也可以是对象
3. 各种特殊装饰器，本质都要理解为嵌套函数的传递。
'''

# 普通装饰器，本身是一个嵌套函数，传入一个原函数


# 特殊装饰器1：本身是一个类的对象的方法，传入一个类  (注册类或者注册函数Registry()应该是一个很流行的设计方法)
import torch.nn as nn
class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
    def _register_module(self, module_class):
        if not issubclass(module_class, nn.Module):
            raise TypeError(
                'module must be a child of nn.Module, but got {}'.format(
                    type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class
    def register_module(self, cls):  # 装饰器函数：传入一个类cls，返回一个类cls
        self._register_module(cls)
        return cls
    
    # 下面这段是更简洁的写法：传入
#    def register_module(self, class_type):
#        module_name = class_type.__name__
#        self._module_dict[module_name] = class_type
#        return class_type
        
backbones = Registry('backbones')  # 创建一个Registry对象
@backbones.register_module         # 挂一个装饰器：用对象的方法作为装饰器，传入的是一个类名，比如ResNet
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
    def forwrad(self):
        pass
model = ResNet()

# 特殊装饰器2：类装饰器，本身是一个类，传入一个原函数
class Deco():
    def __init__(self, func):
        print('func name is {}'.format(func.__name__))
        self._func = func
    def __call__(self):
        print('this is class decorator with new function')
        self._func()
        
@Deco        
def ori():
    print('this is original func')

ori()  # 

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


"""----------------------------------------------------------------
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


"""-----------------------------------------------------------------------
Q. 如何设计带参装饰器？
- 带参装饰器可等效于：func(*dargs,**dkwargs)(f)
"""
def func(*dargs, **dkwargs):    # 带参装饰器的参数
    def deco(f):                    # 传入原函数
        def wrapper(*args, **kwargs):  # 传入原函数参数
            print('this is dec parameter {} and {}'.format(dargs, dkwargs))
            print('this is wrapper parameter {} and {}'.format(args, kwargs))
            return f(*args, **kwargs)
        return wrapper      # 需要2层函数返回
    return deco             # 需要2层函数返回
    
@func(1,a=2)            # 带参装饰器，等效于：test = func(1,a=2)(test)
def test(m, n =10):
    return m + n

test(20, n=10)

# 这种带参装饰器可以参考函数方法链
class add(int):
    def __call__(self, n):
        return add(self+n)
add()(2)(3)  # 第括号是class的实例化，第二个括号是__call__函数形成的函数调用模式的形参，第三个括号是返回的加法函数的参数。
             # 带参装饰器也类似于此

"""-------------------------------------------------------------------------
Q. 如何设计类装饰器，有什么用？
1. 类装饰器是用类来作为装饰器，封装函数，增加额外功能。本质跟函数装饰器相同，
2. 类装饰器的__init__相当于函数装饰器的外层，用来获得函数名称
   类装饰器的__call__相当于函数装饰器的wraper，用来获得函数参数，实现函数调用，以及增加额外功能。
3. 类装饰器通过init获得函数，通过call调用函数。
"""
class Deco():
    def __init__(self, func):
        print('func name is {}'.format(func.__name__))
        self._func = func
    def __call__(self, *args, **kwargs):
        print('this is class decorator with new function')
        return self._func(*args, **kwargs)
        
@Deco        # 等效于ori = Deco(ori), ori(a,b) = Deco(ori)(a,b)
def ori(a, b):
    print('this is original func, sum = {}'.format(a+b))

ori(1,3)  # 等效于调用了Deco(ori)(), 其中Deco(ori)是类装饰器的初始化，调用了__init__
         # 然后初始化后的对象执行__call__方法，Deco(ori)()




"""-------------------------------------------------------------------------
Q. 如何对类的方法添加装饰器？
- 相当与需要装饰器传入参数为对象，此时形参用me_instance代替

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
