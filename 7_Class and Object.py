#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:30:22 2018

@author: suliang
"""

# Class and Object

'''
Q: 类和对象的基本概念？
1. 类的写法，class Dog(): 
    - 定义类，首字母大写，括号内为空，
    - 定义类的实例，首字母小写
2. 初始化方法， __init__(self, name, age):
    - init函数每次创建新实例时，都会自动运行
    - 形参self必不可少，且必须位于其他参数之前，self会自动传递，调用是不需要给出
    - 属性self.name和self.age，带有self的变量叫属性，可供类的所有方法使用，也可供类的所有实例访问
    - 
3. 类的属性

4. 类的方法

'''


'''
Q: 如何定义一个最基本的类和实例？
'''
class Car():                                 # 定义类名：大写，空括号
    def __init__(self, make, model, year):   # 定义初始化函数：必须有self
        self.make = make                     # 类的形参都要写在init()函数形参上
        self.model = model                   # 所有形参都需要内化为属性吗？
        self.year = year
        self.odometer_reading = 0
    
    def read_odometer(self):                 # 定义方法：必须有self？
        print('this car has ' + str(self.odometer_reading) + ' miles on it')
        
    def update_odometer(self, mileage):
        if mileage >= self.odometer_reading:  # 此处用条件禁止改小属性值，但还是可以直接修改属性值
                                              # 比如 new_car.odometer_reading = 15是可以的
            self.odometer_reading = mileage   # 是否还有别的方法可以禁止修改属性？
        else:
            print('You can not roll back an odometer!')
        
new_car = Car('audi', 'a4', 2016)    # 新建一个基于类的实例：类相当于函数，实例相当于函数返回值
new_car.read_odometer()

new_car.update_odometer(200)
new_car.read_odometer()

new_car.update_odometer(15)

        
'''
Q: 如何通过继承一个父类来创建一个子类和实例？
'''
class Ecar(Car):         # 定义一个子类：大写，括号内写父类名字
    def __init__(self,make, model, year):
        super().__init__(make, model, year)  # 用super().来继承父类的初始化函数
        self.battery_size = 70
        
    def describe_battery(self):
        print('this car has a {}kw battery'.format(self.battery_size))

new_Ecar = Ecar('tesla', 'model s', '2016')
new_Ecar.describe_battery()


'''
Q: 如何？
'''
    