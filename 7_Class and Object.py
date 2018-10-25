#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:30:22 2018

@author: suliang
"""

# Class and Object

# 类是什么？类是一套铠甲，穿上它你就有了攻击力计数器(属性)，防御力计数器(属性)，
# 以及强大的攻击方法(方法)，防守魔法(方法)。
# 对象是什么？就是穿着铠甲的勇士

'''
Q: 类和对象的基本概念？
1. 类的写法，class Dog(): 
    - 定义类，首字母大写，括号内为空 ，加冒号(如果没有继承也没有参数，则括号可以不写，但冒号一定有)
    - 定义类的实例，首字母小写
    
2. 初始化方法， __init__(self, name, age):
    - init函数每次创建新实例时，都会自动运行
    - 形参self必不可少，且必须位于其他参数之前，self会自动传递，调用是不需要给出
    - init函数所带的形参，是必须在创建实例时传入类的。

3. 类的属性声明
    - 可通过在init中声明属性self.x=x
    - 属性相当于全局变量，能够提供给类的所有方法使用

4. 类的方法声明
    - 可通过def aaa(self, x)来声明新的方法
    - 新方法的形参只有在调用该方法时才需要传入，init类时不需要传入
    - 新方法的形参只能在该方法内部使用，不具备全局特性。
'''


'''
Q: 如何定义一个最基本的类和方法，以及定义一个实例？
'''
class Car():                                 # 定义类名：大写，空括号
    def __init__(self, make, model, year):   # 定义初始化函数：必须有self
        self.make = make                     # 写在init()函数的形参在定义对象时都需要输入
        self.model = model                   # 内化为属性的形参可以share给class中所有函数使用
        self.year = year
        self.odometer_reading = 0
    
    def read_odometer(self):
        print('this car has ' + str(self.odometer_reading) + ' miles on it')
        
    def update_odometer(self, mileage):
        if mileage >= self.odometer_reading:  # 此处用逻辑禁止改小属性值，但还是可以直接修改属性值
                                              # 比如 new_car.odometer_reading = 15是可以的
                                              # 是否还有别的方法可以禁止修改属性？
            odometer_reading = mileage
        else:
            print('You can not roll back an odometer!')
        self.odometer_reading = mileage

new_car = Car('audi', 'a4', 2016)
new_car.read_odometer()

new_car.update_odometer(200)
new_car.read_odometer()

new_car.update_odometer(15)

        
'''
Q: 如何通过继承一个父类来创建一个子类和实例？
'''
class Ecar(Car):         # 定义一个子类：大写，括号内写父类名字
    def __init__(self,make, model, year):
        super().__init__(make, model, year)  # 继承父类的init方法
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
    def __repr__(self):          # 定义一个__repr__函数，定义实例p的输出显示
        return 'Pair({0.x!r}, {0.y!r})'.format(self)
    def __str__(self):           # 定义一个__str__函数，定义print(p)的输出显示
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
- __getitem__()用于返回一条数据或一个样本： obj[index]等价于obj.__getitem__(index)
- __len__()用于返回样本数量：len(obj)等价于obj.__len__()

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

# 实例参考python cookbook - 类对象一章
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

            
        
    