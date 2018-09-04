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
    - 属性self.name和self.age，带有self的变量叫属性，可供类的所有方法使用，也可供类的所有实例访问
    - 
'''


'''
Q: 如何定义一个最基本的类和方法，以及定义一个实例？
'''
class Car():                                 # 定义类名：大写，空括号
    def __init__(self, make, model, year):   # 定义初始化函数：必须有self
        self.make = make                     # 类的形参都要写在init()函数形参上
        self.model = model                   # 所有形参都需要内化为属性吗？
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
        super().__init__(make, model, year)  # 可与
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
Q: 如何在自定义的类中来定义对象的打印输出？
'''      
        
        
        
        
    