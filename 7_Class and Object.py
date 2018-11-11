#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:30:22 2018

@author: suliang

一些特殊的类方法：
__str__()    用于把实例转化为一个字符串
__repr__()   用于返回实例的代码表示，比如某实例s = Student(10,5), s的代码表示就是Student(10,5)
__format__() 用于把对象按照一定格式输出，比如对象d = Date(), format(d)就能输出自定义格式
def _aaa():  单下划线开头的函数，认为是内部实现，不允许外部调用
_a           但下划线开头的变量，认为是内部变量
def __private(): 双下划线开头的函数或者属性，会产生名称重整，变为_className__priveate_method，并且不能通过继承而覆盖(因为类名不一样了)
lamba_       单下划线结尾的属性，是为了防止与内部保留字冲突(这是基本约定：_开头是内部实现，_结尾是避免名字冲突)

"""

# Class and Object

# 类是什么？类是一套铠甲，穿上它你就有了攻击力计数器(属性)，防御力计数器(属性)，
# 以及强大的攻击方法(方法)，防守魔法(方法)。
# 对象是什么？就是穿着铠甲的勇士

'''
Q: 如何定义一个最基本的类和方法，以及定义一个实例？
'''
class Car():                                 # 类名：后边加不加括号都可以,只有继承类必须加。 class A = class A() = class A(object), 早期python2是要求显式写成A(object)
    def __init__(self, make, model, year):   # __init__()函数不是必须有，他的作用是在初始化一个实例时自动调用，所以通常把形参共享放在init里边做
        self.make = make                     # 写在init()函数的形参在定义对象时都必须要输入
        self.model = model                   
        self.year = year                     # self.x内化为属性的形参可以share给class中所有函数使用,否则仅限本函数使用
        self.odometer_reading = 0            # 所有方法函数，第一个形参必须是self, 包括init函数，但在调用时都不用写
    
    def read_odometer(self):                 # 其他所有函数也必须显式用self为第一个形参。
        print('this car has ' + str(self.odometer_reading) + ' miles on it')
        
    def update_odometer(self, mileage):
        if mileage >= self.odometer_reading:  # 此处用逻辑禁止改小属性值，但还是可以直接修改属性值
                                              # 比如 new_car.odometer_reading = 15是可以的
            self.odometer_reading = mileage
        else:
            print('You can not roll back an odometer!')
        self.odometer_reading = mileage

new_car = Car('audi', 'a4', 2016)            # 创建对象用类名加类初始化形参a=A(x,y)
new_car.read_odometer()                      # 调用类方法： a.method()

new_car.update_odometer(200)                 # 调用带参方法
new_car.read_odometer()

new_car.update_odometer(15)



'''
Q: 在类的体内如何引用类的其他子函数，如何引用类的其他子函数的变量？
'''
# 定义类内部的共享变量：self.a = 6

# 调用类内部其他变量: b = self.a

# 调用类内部其他函数：b = self.func1()

        
'''
Q: 如何通过继承一个父类来创建一个子类和实例？
'''
class Ecar(Car):                            # 继承一个类：首字母大写，括号内写父类名字
    def __init__(self,make, model, year):
        super().__init__(make, model, year)  # super()函数用于获得父类名称使用权，然后调用父类init()。由于是调用，所以init里边不写self，但要写父类初始化的相关形参
        self.battery_size = 70
        
    def describe_battery(self):
        print('this car has a {}kw battery'.format(self.battery_size))

new_Ecar = Ecar('tesla', 'model s', '2016')
new_Ecar.describe_battery()


'''
Q: 如何在自定义的类中来定义对象的打印输出？
'''
class Pair():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):          # 定义一个__repr__函数，定义实例p的输出显示，即p的显示
        return 'Pair({0.x!r}, {0.y!r})'.format(self)
    def __str__(self):           # 定义一个__str__函数，定义实例p的打印输出显示，即print(p)的输出显示
        return '({0.x!s}, {0.x!s})'.format(self)
# 一个编写完善的类，一般会定义__repr__和__str__用来给用户和程序员提供更多关于实例的信息
# 这两个函数都能在特定情况下自动调用

p = Pair(3,4)
print(p)   # 调用的是__str__
p          # 调用的是__repr__


'''
Q: 如何自定义一个数据类？
- 自定义一个数据类
- 重写__getitem__()函数和__len__()函数
- __getitem__()用于返回一条数据或一个样本： 在切片调用时自动调用，obj[index]等价于obj.__getitem__(index)
- __len__()用于返回样本数量：在求len()时自动调用，len(obj)等价于obj.__len__()

'''
# 本数据类参考陈云的pytorch教程第5章
import os
from PIL import  Image
import numpy as np

# 最简版的一个数据类，但涵盖了数据类定义的基本要点
#
class DogCat(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)  #获得每张图片地址
        self.imgs = [os.path.join(root, img) for img in imgs]
    # 把数据提取的内容都放到__getitem__来实现，
    # 只有在每张图片需要加载时才会调用该函数，且多进程(多图片)可以并行
    # 也就不用一次性把所有图片都加载到内存去    
    def __getitem__(self, index):  
        img_path = self.imgs[index]
        # dog->1， cat->0
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        data = t.from_numpy(array)
        return data, label
    
    def __len__(self):
        return len(self.imgs)

        
        
'''
Q: 如何使用getattr和setattr方法实现调用对象上的方法和方法名用字符串形式给出？
（参考python cookbook 8.20）
1. getattr的用法：getattr(obj. , attr.)等效于obj.attr
    getattr(obj, 'name')  获得对象的name属性，存在就打印出来
    getattr(obj, 'run')   获得对象的run方法，存在就打印方法的内存地址
    getattr(obj, 'run')() 获得方法，并且运行该方法
    getattr(obj, 'age','18') 想要获得一个不存在的方法，但预设了不存在的默认返回值。
   另一个__getattr__()方法，对于obj.attr的指令，系统对默认先找getattr()函数，如果没有则调用obj.__getattr__()兜底，如果也没有则报错AttributeError 

2. setattr的用法：为对象已有属性赋值，或者创建新属性
    setattr(obj, 'age', '18') 如果age属性存在，则赋值为18,如果不存在则先创建再赋值为18
    对于该命令，系统默认调用__setattr__()函数
'''

# 实例参考python cookbook8.2
import math
class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return 'Point({!r:}, {!r:})'.format(self.x, self.y)
    def distance(self, x, y):
        return math.hypot(self.x - x, self.y - y)

p = Point(2,3)

dist = getattr(p, 'distance')(0,0)  # 相当与p.distance(0,0), 但可以采用字符串形式给到getattr(),在某些场合有这种需求
print(dist)

            
'''
Q: 如何对一个类的属性进行包装，增加更多判断和检查？
（参考python cookbook 8.8）

核心理解：一个属性aa定义好以后，其实默认有3个方法为他服务
aa.getter()方法，在输出属性时会自动调用他
aa.setter()方法，在设置属性时会自动调用他
aa.deleter()方法，在删除属性时会自动调用他

@property本质就是定义getter()方法，只要定义了getter()方法，就能直接调用
比如一个方法bbb()之前增加@property，就能obj.aaa这样调用，相当于属性化一个函数

''' 
# 对一个函数添加@property,相当与把函数转化为property(属性)，从而可以直接用p.first_name来调用
# 添加了@property后，根据函数输入的不同，会自动触发getter, setter, deleter方法
# 
class Person:
    def __init__(self, first_name):
        self.first_name = first_name
    
    @property             # getter方法，必选
    def first_name(self):
        return self._first_name
    
    @first_name.setter     # setter方法，可选
    def first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value
    
    @first_name.deleter   # deleter方法，可选
    def first_name(self):
        raise AttributeError('Can not delete attribute')

p = Person('leo')        
p.first_name           # 显示的是属性
p.first_name = 42      # 调用的是属性同名函数的setter方法，因为输入数字，所以报错
p.first_name = 'martin'   # 正常
del p.first_name       # 调用的是属性同名函数的deleter方法

 
'''
Q: 如何把一个类里边几个简单的计算函数统一成类的属性，便于使用？
''' 
# 使用单独的@property方法
import math
class Circle:
    def __init__(self, radius):
        self.radius = radius
    @property  # 把area函数属性化, 相当与调用getter函数
    def area(self):
        return math.pi * self.radius**2
    @property  # 把perimeter函数属性化
    def perimeter(self):
        return 2* math.pi*self.radius

c = Circle(10)
c.radius
c.area     # 通过添加@property，访问area就像访问radius一样，不用加括号
c.perimeter


'''
Q: 如何调用父类中已经被子类覆盖的方法？
- 注意常见python2与python3在super()使用区别：
    - super(B, self).Parent_method()  # 这是python2的写法
    - super().Parent_method()         # 这是python3的写法，两个都是对的
''' 
# 使用super()函数，相当与获得了父类名称的使用权
class A:
    def spam(self):
        print('A.spam')
        
class B(A):
    def spam(self):
        print('B.spam')
        super().spam()
a = A()
a.spam()
b = B()
b.spam()

# super()还常用在继承父类时的init方法上
class A:
    def __init__(self):
        self.x=0
    
class B(A):
    def __init__(self):
        super().__init__()
        self.y = 1