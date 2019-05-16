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
Q: 如何完整理解父类和子类的关系？
参考：https://www.cnblogs.com/cccy0/p/9040192.html, 写得比较完整
% 初始化      类不能没有__init__，子类没有__init__则自动调用父类__init__
              如果子类写了自己__init__，则不会init父类，要使用父类属性就需要手动
              调用父类super().__init__()
% 调用属性     子类可以调用父类属性，但需确保父类属性初始化，
               同时不同名可直接调用，同名则需要通过super()调用
% 调用方法     子类可以调用父类方法，不同名直接调用，同名则通过super()调用
'''
class Animal():     # 父类
    def __init__(self):
        self.head = 1
    def drink(self):
        print('Animal is drinking!')
        
class Rabbit(Animal):  # 子类
    pass

rabbit = Rabbit()
rabbit.drink()                                  # 关键1：子类可随意调用父类方法
print('it has {} head'.format(rabbit.head))     # 关键2：子类可随意调用父类属性

class Chick(Animal):
    def __init__(self):
        super().__init__()   # 关键5，要使用父类属性就要确保父类初始化
        self.legs = 4
    def drink(self):
        print('Chick is drinking!')
    def parentdrink(self):
        super().drink()
        
chick = Chick()
chick.drink()               # 关键3： 子类调用自己的方法(与父类同名方法)，可随便调用
chick.parentdrink()         # 关键4： 子类调用父类同名方法，需要用super()

                       # 关键5： 子类调用父类属性需要确保父类init
                       #         如果子类没有自己__init__则自动调用父类__init__()完成初始化
                       #         如果子类有自己的__init__则需要手动调用父类super.__init__()
print('it has {} head, and chick has {}legs'.format(chick.head, chick.legs))



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
__repr__()方法：在实例输出时自动调用，如obj，往往用来放入可以eval()的代码片段
__str__()方法：在实例打印输出时自动调用，如print(obj)，往往用来放入对实例的基本描述
'''
class Pair():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):         
        return 'Pair({0.x!r}, {0.y!r})'.format(self)
    def __str__(self):           
        return '({0.x!s}, {0.x!s})'.format(self)


p = Pair(3,4)
print(p)   # 调用的是__str__
p          # 调用的是__repr__


'''
Q: 如何自定义一个数据类？
- 自定义一个数据类
- 重写__getitem__()函数和__len__()函数
- __getitem__()用于返回一条数据或一个样本，在切片调用时自动调用，obj[index]等价于obj.__getitem__(index)
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
# 同样应用案例
for hook in hooks:
    getattr(hook, 'before_epoch')(self)


# %%
"""基于子类可以获得父类的信息，那如何基于父类来得到子类？比如基于torch.optim父类，得到子类torch.optim.SGD？
"""    
import torch.optim
type = 'SGD'
new = getattr(torch.optim, type)




# %%
'''
Q. 如何定义类的三种方法：类方法@classmethod，静态方法@staticmethod，属性方法@property
三者的区别和使用范围是什么？
关键0：理解类和对象的关系
    * 类相当于出厂就能动的最小系统机器人，有最小系统下的类属性和类方法
      而对象相当于已通电全功能版机器人，不仅有类属性/类方法，还增加对象属性和普通方法
    * 对象是类实例化以后的产物，拥有了自己的属性和方法，叫对象属性和普通方法
    * 可视化直观理解：类是机器人大脑，类方法只能访问类属性，相当与机器人大脑有一圈虚线一圈实线，只能从对象往类单向访问；
      对象是机器人全身，普通方法可以访问对象和类的所有属性
      静态方法相当与机器人的虚线尾巴，不能访问机器人的类和对象，只是挂在机器人身上而已

关键1：区分类属性和对象属性
    * 类属性：属于类本身的属性，不会改变(相当于核心变量)
    * 对象属性：属于对象的属性，可以改变
    
关键2：区分方法产生目的和调用方式
    * 静态方法最自由，可以脱离类和对象运行，被类和实例调用
      (相当与一个独立函数，只是被类和对象来调用)
    * 类方法也自由，可以脱离对象运行，被类和实例调用
    * 普通方法只能被实例调用
      属性方法属于普通方法，所以也只能被实例调用

关键3：区分3大方法   
     * 类方法，classmethod，必写的隐含参数是cls，不可以访问对象属性，可以被类和对象调用
     * 普通方法(也叫实例方法)，必写的隐含参数是self，可以访问任何属性，只可以被对象调用
         * 属性方法属于普通方法的一种，所以必写隐含参数self
     * 静态方法，staticmethod，没有必写隐含参数(即可以不写self或cls, 也可以写self/cls用来调用对象属性)，
       不写self则不可以访问类或对象的任何属性/方法，但可以被类和对象调用
       但写了self就是传入对象self作为一个形参, 然后就可以调用对象属性self.attri
       也可传入类cls，然后调用类属性

关键4：几种特殊方法的典型应用
     * @property属性方法：是普通方法的一种，主要用来把方法属性化，调用方便省去写括号，a.action
     * @classmethod类方法：用来获得类的一些基本属性展示
     * @staticmethod静态方法：用来剥离出来放一些独立逻辑，跟类和对象都不会交互，但可以被大家使用。
       
'''
#----------普通方法案例-------------------------
class Dog():
    def __init__(self, name):
        self.name = name
        self.__food = None
    def eat(self, food):   # 普通方法
        print('%s is eating %s'% (self.name, food))
d = Dog('Jack') 
d.eat('pie')    # 普通方法被对象调用
#----------静态方法案例-------------------------
class Dog():
    default_name = 'David'
    def __init__(self, name):
        self.name = name
        self.__food = None
    
    @staticmethod
    def eat(self):  # 静态方法，参数为对象
        print('%s is eating %s'% (self.name, 'pie'))
        
    @staticmethod
    def run(name): # 静态方法，参数为形参
        print('%s is running!'% name)
    
    def jump(cls):  # 静态方法，参数为类
        print('%s is running!'% cls.default_name)
        
d = Dog('Jack')
Dog.eat(d)         # 类调用静态方法，传入对象
d.eat(d)           # 对象调用静态方法，传入对象
Dog.run('Jack')    # 类调用静态方法，传入新参数
d.run(d.name)      # 对象调用静态方法，传入对象属性
Dog.run(Dog.default_name)  # 类调用静态方法，传入类属性
Dog.jump(Dog)              # 类调用静态方法，传入类
# ----------类方法案例-----------------------
class Dog():
    food = 'bone'
    def __init__(self, name):
        self.name = name
        self.__food = None
    @classmethod
    def like(cls, food):   # 普通方法
        print('dogs like %s'% (food))

Dog.like('food')    # 类调用类方法，传入额外参数
Dog.like(Dog.food)  # 类调用类方法，传入类属性

d = Dog('Jack')
d.food = 'pie'
print(d.food, Dog.food)   # 可以发现，类属性没有被对象改变，只是类属性和对象属性同名

d.like('food')      # 对象调用类方法，传入参数'food'，而不是获取类属性
d.like(d.food)      # 对象调用类方法，传入对象属性
d.like(Dog.food)    # 对象调用类方法，传入类属性
# ----------属性方法案例-----------------------
class Dog():
    def __init__(self, name):
        self.name = name
        self.food = None
    @property
    def like(self):   # 添加属性方法，可以不用写括号
        print('dogs like %s'% (self.food))
                      # 属性方法优点是不用写括号，缺点是不能像真正的属性一样赋值。
    @like.setter      # 但可以通过增加
    def like(self, food):
        self.food = food
    @like.deleter
    def like(self,food):
        del self.food
a = Dog('David')
a.like = 'pie'  # 调用属性方法的setter函数
a.like          # 调用属性方法
del a.like      # 删除属性方法


'''
Q: 如何理解和使用抽象类和抽象方法@abstractmethod?
'''
from abc import ABCMeta, abstractmethod
class Person():
    __metaclass__ = ABCMeta  # 定义了抽象类
    def name(self):
        pass



'''
Q: 如何对一个类的属性进行包装，增加更多判断和检查？
（参考python cookbook 8.8）

关键1： 充分理解对象的属性在实现时的调用过程
要操作一个属性，实际上背后是调用三个函数
获得属性值：相当于调用getter()函数，在输出属性时会自动调用他
设置属性值：相当于调用setter()方法，在设置属性时会自动调用他
删除属性值：相当于调用deleter()方法，在删除属性时会自动调用他

关键2：理解@property本质就是定义getter()方法，所以@property之后只能获得属性。
要想对这个属性进行设置和删除操作，需要额外定义@xx.setter(), @xx.deleter()
''' 
# 对一个函数添加@property,相当与把函数转化为property(属性)，从而可以直接用p.first_name来调用

class Person:
    def __init__(self, first_name):
        self.first_name = first_name
    
    @property             # 有这句@property就相当于定义了getter()函数
    def first_name(self):
        return self._first_name
    
    @first_name.setter     # 需要自己定义setter()函数
    def first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value
    
    @first_name.deleter   # 需要自己定义deleter()函数
    def first_name(self):
        raise AttributeError('Can not delete attribute')

p = Person('leo')        
p.first_name           # 显示的是属性
p.first_name = 42      # 调用的是属性同名函数的setter方法，因为输入数字，所以报错
p.first_name = 'martin'   # 正常
del p.first_name       # 调用的是属性同名函数的deleter方法


'''
Q. 如何使用类和对象的__dict__方法？
核心理解1：__dict__是存放所有属性的地方，python中所有对象都有__dict__属性，除了几个内置数据类型int/float/list/dict
核心理解2：在类的__dict__存放了类属性和类方法，而在对象__dict__中只存放对象属性(临时变量是不存放的)
'''
class Dog():
    default_name = 'Alan'
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def who(self, name):
        print('%s is dog default name, but %s is dog new name'% (Dog.default_name, name))
    def like(self, food):
        print('%s likes %s'% (self.name, food))

d = Dog('David', 8)        
print(Dog.__dict__)
print(d.__dict__)

 
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
# 使用super()函数，相当与获得了父类名称的使用权，
# 即super() = A, 
#   super().__init__() = A.__init__()
#   super().spam() = A.spam()
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
        
        
'''
Q. 如何使用__new__()方法
    该方法用于指定一个类来创建实例，需要返回该实例
    实例参考：http://www.cnblogs.com/ifantastic/p/3175735.html
    
关键点：
    充分理解创建一个对象的内部调用过程：
    * 先调用__new__()方法，基于传入的类来创建对象
    * 再调用__init__()函数，初始化相关变量
    可见真正创建对象是由__new__()来完成，而不是由__init__()来完成
'''

class AAA(object):
    def __init__(self, *args, **kwargs):
        print('foo')
    def __new__(cls, *args, **kwargs):  # 传入的是一个类
        return object.__new__(Stranger, *args, **kwargs)  

class Stranger(object):
    def __init__(self, *args, **kwargs):
        print('stranger')

a = AAA()
print(type(a))    # 可以发现明明是AAA的类，但实际上生成的实例是Stranger类的实例



'''
Q. 如何使用__setattr__()和__getattr__()方法？
    __getattr__()方法：在获得属性时调用，比如a.name, 就调用__getattr__(self,name)
                       先从self.__dict__中搜索，在init函数添加的属性都自动
                       在__dict__中，如果没找找到再从__getattr__()方法中搜索
    __setattr__()方法：在设置属性时调用，比如a.name='Eason'，就调用__setattr__(self, name, 'Eason')
                       所以在初始化一个实例时，就会调用__setattr__方法把参数加到__dict__里边去
'''    
class A:  
    def __init__(self):  
        self.gendor = 'male'  
  
    def __getattr__(self, item):  
        '''attr的获得方式有2条：先从self.__dict__中搜索，在init函数添加的属性都
        自动在__dict__中，然后从__getattr__()方法函数中搜索。本例中gendor是从
        __dict__找到的，name/age是从__getattr__找到的
        '''
        print('this is __getattr__()')
        if item == 'name':  
            return 'xyz'  
        elif item == 'age':  
            return 26 
        
    def __setattr__(self, name, value):
        print('this is __setattr__')
        return super().__setattr__(name, value)

a = A()          # 初始化时，通过__setattr__方法把属性加入self.__dict__
print(a.gendor)  # 搜索a.__dict__的到，不调用__getattr__()
print(a.name)    # 先搜索self.__dict__，没搜索到，然后调用__getattr__()搜索到
print(a.age)    

a.school = 'taoliyuan'    
a.sub_module = [1,2]    # 支持添加一个属性值为list的属性

a.__dict__    



'''
Q. 区分__getattr__()和__getattribute__()方法？
    __getattribute__()是新类的方法，每次获得属性都会调用该方法
    __getattr__()只有在__dict__里边没有该属性才会调用该方法
'''  
# 还以这个类为例
class A:  
    def __init__(self):  
        self.gendor = 'male'  
  
    def __getattr__(self, item):  
        print('this is __getattr__()')
        if item == 'name':  
            return 'xyz'  
        elif item == 'age':  
            return 26 
        
    def __setattr__(self, name, value):
        print('this is __setattr__')
        return super().__setattr__(name, value)
    
#    def __getattribute__(self, item):
#        print('this is __getattribute__')
#        print('you visited:', item)
#        return super().__getattribute__(self,item)

a = A()
print(a.gendor)  # 该条没有调用__getattr__()



'''
Q. 区分__getattr__()和getattr()方法？
1. 关键1：getattr()是直接访问__dict__去查找属性，如果没有找到才会调用__getattr__()继续查找
2. 关键2：getattr()函数可以设置属性不存在的默认返回值而不会报错
'''  
class A:  
    def __init__(self):  
        self.gendor = 'male'  
  
    def __getattr__(self, item):  
        print('this is __getattr__()')
        if item == 'name':  
            return 'xyz'  
        elif item == 'age':  
            return 26 
        
    def __setattr__(self, name, value):
        print('this is __setattr__')
        return super().__setattr__(name, value)
    
    def printit(self):
        print('this is print')

a = A()   # 调用__setter__()
a.gendor  # 先查看__dict__有，所以没有调用__getattr__()
a.apple   # 先查看__dict__没有，所以调用__getattr__()
print(getattr(a, 'gendor', 'not exist'))   # getattr()函数可以获得已有属性
print(getattr(a, 'school', 'not exist'))   # __dict__没有所以调用__getattr__查看
getattr(a,'printit')()                    # getattr()函数可以获得已有方法
setattr(a,'book',5)
getattr(a,'book')


'''
Q. 如何使用__call__()方法？
核心理解1：__call__方法可以把类的对象像函数一样调用，
核心理解2：__call__只针对对象，所以__call__之前肯定先会实例化
    所以需要区分call和init，一个是针对对象(call)，一个是针对类(init)
'''  
import torch.nn as nn
class Registry(object):
    def __init__(self):
        print('this is init')
        self.module_dict={}
    
    def __call__(self, class_type):
        print('this is call')
        module_name = class_type.__name__
        self.module_dict[module_name] = class_type
        return class_type

registry = Registry()        

@registry                  # 这里只能用对象做调用
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
    def forwrad(self):
        pass
print(registry.module_dict)
#model = ResNet()
print(registry.module_dict)


'''
Q. 如何使用__str__()方法？
'''  



'''
Q. 如何使用__iter__？
- 迭代器的特点：
- 如果想要创建一个迭代器用在for循环中，则需要对一个类创建__iter__()和next()方法
'''  
class Fib(object):
    '''创建一个斐波那切迭代器
    '''
    def __init__(self, n):
        self.a, self.b = 0, 1 # 初始化两个计数器a，b
        self.n = n

    def __iter__(self):
        return self      # 实例本身就是迭代对象，故返回自己

    def next(self):
        self.a = self.b       # 计算下一个值
        self.b = self.a + self.b
        if self.a > self.n:   # 退出循环的条件
            raise StopIteration()
        return self.a # 返回下一个值

fib = Fib(10)
for i in fib:
    print(i)


'''-------------------------------------------------------------
Q. 特殊方法之__add__和__radd__
理解1：__add__/__radd__都是用来重载运算符的，就是把+赋予更多功能
理解2：a + b，首先调用a的方法a.__add__(b)，如果a没有重写该方法，则调用b的方法b.__radd__(a)
如果a,b都没有重写这两个方法之一，则报错
'''
class A():
    def __add__(self, x):
        print('A__add__')
    
    def __radd__(self, x):
        print('A__radd__')

class B():
    pass

a = A()
b = B()

a + b  # 先调用a的__add__方法，存在所以输出
b + a  # 先调用b的__add__方法，不存在所以继续调用a的__radd__方法，存在所以输出


'''-------------------------------------------------------------
Q. 特殊属性之__dict__
核心理解1：__dict__是存放所有属性的地方，python中所有对象都有__dict__属性，除了几个内置数据类型int/float/list/dict
核心理解2：在类的__dict__存放了类属性和类方法，而在对象__dict__中只存放对象属性(临时变量是不存放的)
'''
class A():
    j = 1
    k = 2
    def __init__(self):
        self.j = 3
        self.k = 4
        m = 5
        n = 6

a = A()

print(A.__dict__)
print(a.__dict__)


'''-------------------------------------------------------------
Q. 特殊属性之__file__
核心理解1：__file__是存放当前文件名
'''
print(__file__)


'''-------------------------------------------------------------
Q. 特殊方法之getattr()， setattr(), hasattr()以及与__getattr__, __setattr__的区别
getattr()函数本质上是调用对象的__getattr__()方法
setattr()函数本质上是调用对象的__setattr__()方法
'''
class AA():
    name = 'Eason'

# getattr()    
a = AA()
print(getattr(a,'name'))   # 获得某属性，注意属性名需要用引号
print(getattr(a, 'work', 'HelloKitty'))  # 可以设置没有返回缺省值

# setattr()
setattr(a, 'age', 10)
print(getattr(a, 'age'))
print(a.age)

# hasattr()
hasattr(a, 'name')
hasattr(a, 'gender')


'''-------------------------------------------------------------
Q. 特殊类Dict
- 可以实现用访问属性的方式来访问字典，并且支持嵌套字典
'''
from addict import Dict

d1 = Dict(a=1, b=2)
d1.a

d2 = dict(a=2,b=dict(c=3,d=4),e=4)
d2 = Dict(d2)
d2.b.d

d3 = Dict(a=1, b=dict(c=2,d=3),e=5)
d3.b.d


'''------------------------------------------------------------------------
Q. 如何实现多重继承，继承顺序是什么？
1. 经典类和新式类的区别：python3以后，所有类都是隐式继承object，都是新式类
2. 两种常见继承结构：
   >并行继承：如果一个类继承自两个父类，而两个父类分别继承自不相关另两个父类，这种继承体系并行不相干，称为并行继承。
   >钻石继承：如果一个类继承自两个父类，而两个父类又来自同一基类，这种继承体系很像个钻石或者菱形，所以叫钻石继承或者菱形继承。
   新式类由于都是继承自object, 所以任何多重继承都是钻石继承
3. 所有继承的顺序是基于MRO(方法解析顺序)所定义的算法来决定的，其中MRO所使用的算法在python中经历很多变化
   早期mro采用DFS(深度优先)，后来mro改为BFS(广度优先)，现在mro统一成C3算法(不是很多人说的广度优先，只是恰巧在部分情况跟广度优先结果相同)
   比如两种经典结构中，并行继承结构如果用c3算法得到结果跟深度优先一样，菱形继承结构如果用C3算法得到的结果跟广度优先一样。
4. mor采用c3算法主要是为了解决单调性(并行继承时应该单线到底称为单调性)和继承无法重写(钻石继承应该确保平行的另一父类能够被访问)的问题
   C3算法同时避免了DFS和BFS的问题，就像DFS和BFS的合体一样，非常惊艳
5. mro顺序得到后的应用重点：
   >在初始化时，因为super()是按mro顺序寻找父类，所以最终是基于mro相反的顺序初始化类的链条
   >在寻找某方法时，按照mro的顺序从最先找到的父类方法作为执行方法。
   
参考：http://python.jobbole.com/85685/  (讲清楚了DFS/BFS/C3的差异)
https://www.jianshu.com/p/71c14e73c9d9  (提出了init与mro顺序正好相反)
'''
# 并行继承结构
class E():pass
class D():pass
class C(E):pass
class B(D):pass
class A(B,C):pass
A.__mro__   # mro为A,B,D,C,E,O 类似DFS

# 菱形继承结构
class D():pass
class C(D):pass
class B(D):pass
class A(B,C):pass
A.__mro__   # mro为A,B,C,D,O 类似BFS

# 先来一个简单的
class A():
    def __init__(self):
        print('A init')
class B(A):
    def __init__(self):
        super().__init__()
        print('B init')
class C(A):
    def __init__(self):
        super().__init__()
        print('C init')
    def say(self):
        print('say C')
class D(B, C):
    def __init__(self):
        super().__init__()
        print('D init')

D.__mro__   # D的mro顺序是D,B,C,A,O
d = D()     # d init的顺序正好跟mro相反，可以理解为用super()搜索__init__函数，最终先停在A，A执行完回退到C，就这样按照mro相反顺序回退执行__init__
d.say()     # d引用say()方法，但自己没有这个方法，所以按照mro顺序去找(D,B,C,A)，先在C找到了

# 修改上面例子，增加计算
class A():
    def __init__(self):
        self.value = 1
        print('A init, value = {}'.format(self.value))
class B(A):
    def __init__(self):
        super().__init__()
        self.value *= 2
        print('B init, value = {}'.format(self.value))
class C(A):
    def __init__(self):
        super().__init__()
        self.value += 5
        print('C init, value = {}'.format(self.value))
class D(B, C):
    def __init__(self):
        super().__init__()
        print('D init, value = {}'.format(self.value))
# 传统初始化方法基于深度优先，碰到重复类就保留最后一个，则应该是D-B-A-O-C-A-O保留最后一个成为D-B-C-A-O
# 新式类初始化基于C3算法: L[D] = D + [L[B], L[C],B,C] = D + [[B, A, O] + [C, A, O],B,C] 
#                                                    = D + B + C + A + O        
foo = D()
print(foo.value)    
D.__mro__   # D class的mro顺序如同上面计算的D,B,C,A,O, 但为什么显示的打印过程不是这样的？？？

# 另一个更复杂的多重继承
class X(object):
    pass
class Y(object):
    pass
class A(X,Y):
    pass
class B(Y,X):
    pass
class C(A,B):
    pass
c = C()
# 传统深度遍历为C-A-X-O-Y-O-B-Y-O-X-O，保留最后一个变为C-A-B-Y-X-O
# 缺陷是子类居然改变了积累的方法搜索顺序
# 采用C3算法进行线性化计算初始化顺序的过程：知道L的公式，知道merge的过程(依次检查第i个表的第一个元素，不在后端则输出并删除)
# 计算时关键是merge那步：检查第一个列表的第一个元素，如果没出现在其他列表尾部则输出并从全部列表删除，然后重复再来。
# 直到列表为空则完成list构建，直到不能找到可输出的且列表不为空则说明无法构建继承关系则报错。
#    L[O] = O, L[X]=[X,O], 
#    L[C(B1..Bn)]= [C] + merge(L[B1], L[B2].., B1, B2,..)
L[A] = A + merge(L[X], L[Y], X, Y) 
     = A + merge([X, O], [Y, O], X, Y)
     = A + X + Y + O                 # 这个类似广度优先，但实际不是只是恰巧一样
L[B] = B + merge(L[Y], L[X], Y, X) 
     = B + merge([Y, O], [X, O], Y, X)
     = B + Y + X + O
L[C] = C + merge(L[A], L[B])
     = C + merge([A, X, Y, O], [B,Y,X,O], A, B)
     = C + A + merge([X, Y, O], [B,Y,X,O], B)   # 第一个列表的X出现在了第二个列表的后端，非法了，所以跳到第二个列表的头部取B
     = C + A + B + merge([X, Y, O], [Y,X,O])    # 地一个列表的X，第二个列表的Y都非法了，所以报错
     = 无法继续计算，报错

# 计算一个正确的
class X(object):
    pass
class Y(object):
    pass
class A(X,Y):
    pass
class B(X,Y):
    pass
class C(A,B):
    pass
c = C()
C.__mro__
#
L[C] = C + merge(L[A], L[B],A,B) = C + merge([A,X,Y,O], [B,X,Y,O],A,B)
     = C + A + B + X + Y + O



'''-------------------------------------------------------------------------
Q.如何使用Mixin类？
Python是支持多重继承的，但多重集成的问题在于继承关系混乱，有限顺序不明确，相同方法冲突不明确。
所以不同语言采用不同解决方案，java只能有一个父类但多个interface，而python的优化方法就是Mixin方法，
松本行弘的程序世界建议原则是：常规继承采用单一继承，第二个以上的父类采用Mixin抽象类，继承是i am, 而mixin是i can。

具体领域1: 有一些独立的强大方法，希望用来扩展某些类，
具体领域2: 对父类原本的方法不太满意，在Mixin中用super()调用，同时增加辅助功能（有点类似装饰器）

1. Mixin类是一个功能，而不是一类对象。他不能直接实例化，
2. Mixin类责任单一，多个功能就应该多个Mixin类
3. 多重继承的子类即使没有Mixin类，也是能工作，只是少了点功能
4. Mixin类一般没有状态，即没有__init__函数，也没有实例变量。可以通过添加__slots__=()来
5. Mixin类中使用super()函数是必要的，这时super()会在继承的子类中基于MRO方法解析顺序从下一个类中调用

'''
# 基类
class Animal(object):  
    pass
# 大类:
class Mammal(Animal): # 初级子类1
    def like_to_eat(self):
        print('it likes milk!')
class Bird(Animal):  # 初级子类2
    pass
# Mixin类
class RunnableMixin(object):
    def run(self):
        print('Running...')
class FlyableMixin(object):
    def fly(self):
        print('Flying...')
# 新子类
class Dog(RunnableMixin, Mammal):
    pass

d = Dog()
d.run()            # d没有run方法，基于mro先从Mixin类找，获得额外的方法
d.like_to_eat()

# 另一个Mixin的例子from mmdetection.two_stage_detector
class RPNTestMixin(object):
    """这个Mixin类用于被detector类继承，提供一些测试方法"""

    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, img_meta, rpn_test_cfg)
            for proposals, img_meta in zip(aug_proposals, img_metas)]
        return merged_proposals    
    
# 还有一个比较复杂的Mixin例子：https://blog.csdn.net/u012814856/article/details/81355935
class Displayer():
    def display(self, message):
        print('this is Displayer class with message: {}'.format(message))
        

class LoggerMixin(): 
    def log(self, message): 
        print('this is LoggerMixin class with log: {}'.format(message))       
    def display(self, message):
        super().display(message)
        self.log(message)
        
class MySubClass(LoggerMixin, Displayer):
    def log(self, message):
        print('this is MySubClass with log: {}'.format(message))
        
subclass = MySubClass()
subclass.display("this is a message")   # 先找到父类LoggerMixin, 通过super()到Displayer类的display()函数
                                        # 然后执行父类loggerMixin的self.log()函数，但此时的self是MySubClass而不是LoggerMixin，所以是调用MySubClass的log()


# %%
"""如何实现对象的工厂模式？
1. 需要先获得一组类的列表
2. 从类列表中提取类名
3. 根据类名生成对象
"""
# 方式1：通过导入package来获得这个package下面所有的类，但需要先把类导入package的__init__文件的__all__变量中
from .. import datasets
args = dict(a=1)                          # 这里datasets是一个package(也就是一个文件夹)
obj_type = getattr(datasets, class_name)  # 根据类名获得类
obj = obj_type(**args)                    # 生成类的对象

# 方式2：通过registry()装饰器函数，预先搜集所有类

