#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:29:54 2018

@author: suliang
"""

# functions


'''-------------------------------------------------------------------------
Q: 怎么编写可接受任意数量的位置参数，或者任意数量的关键字参数？
精彩参考：https://www.cnblogs.com/bingabcd/p/6671368.html
1. 核心概念：
    核心是2大类，位置参数(同类参数用位置顺序区分，可以不写变量名传参)，
    关键字参数(必须用变量名+等号传参,同类参数没有顺序，用变量名区分)
    (1)位置参数def foo(x,y)
    (2)默认参数def foo(x=1)， 注意默认参数是位置参数的一种形参表现形式，但他不是关键字参数
    这也是为什么可变位置参数*arg与默认参数的相对位置关系可以交换
    可以这么理解：形参上只能写 def foo(a,b=2,**kwargs)，没有其他关键字参数的写法；
    而实参上写foo(1,b=3,c=3,d=4)，则默认参数与关键字参数形式上相同，通过变量名区分
    (3)可变位置参数def foo(*args)
    (4)关键字参数 def foo(*, a) 此时*强制要求之后的变量为关键字参数变量
    (5)可变关键字参数def foo(**kwargs)
2. 本质
    (1)*args：其中args代表元组，*代表拆包操作，*args代表拆包完成的多个位置参数
        例如：args = (1,2,3)，则*args -> 1,2,3
    (2)**kwargs：其中kwargs代表字典，**代表拆字典操作，**kwargs代表拆字典完成的多个关键字参数
        例如：kwargs={'a':1,'b':2}, 则**kwargs -> a=1, b=2
3. 形参顺序定义：
    (1)位置参数 -> *args -> 默认参数 -> **kwargs
    (2)位置参数 -> 默认参数 -> *args -> **kwargs  (即默认参数与*args可变位置参数可以换，其他固定)
    (3)在*作为分割定义的为关键字参数比如 def foo(a,*args, b, **kwargs)其中b就是关键字参数而不是位置参数，传入必须加等号
4. 实参识别模式
    (1)形参定义时除了在*之后和**kwargs被强制定义成关键字参数，其他参数默认都是位置参数/默认参数
    (2)实参传入时，无论位置参数/默认参数/关键字参数，都可以用等号传入。区别在于位置参数/默认参数还可以直接不带变量名基于位置写数值

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


# 一个核心应用: 用*可变位置参数或**可变关键字参数，在函数体运行时传参进去
# 这样就可以实现从config文件读出turple或者dict，然后通过*/**解包之后直接用作函数实参
def avg(x=1,y=1,z=1):
    s= x + y + z
    return s

data = dict(x=1,y=2,z=3)  # 事先可以在config中准备这些信息
avg(**data)       # 调用函数时可以很方便的解包参数后作为形参，调用过程代码就很简洁
avg(1,1,1)

# 区分默认参数与关键字参数
def foo(a, b=2, **kwargs):     # 形参上的区别：默认参数用=等号的写法，而关键字参数只能有**kwargs的方式，没有等号写法
    print('position arg a: {}'.format(a))
    print('moren arg b: {}'.format(b))
    print('keyword args: {}'.format(kwargs))
foo(1, b=3, c=4, d=5)         # 实参上：默认参数与关键字参数写法相同，都用等号写法，通过变量名来区分

# 四种参数的位置区别
def foo(x, *args, a=4, **kwargs): # 默认参数的位置在args之后kwargs之前，此时默认参数的调整需要带名称调整，否则会被位置参数先吃掉
    print(x)
    print(a)
    print(args)
    print(kwargs)
foo(1,2,3,4,5,a=5)  # 此时如果不写a=5,则所有数字都被排在前面的默认参数吃掉

def foo(x, a=4, *args, **kwargs): # 默认参数的位置在args之后kwargs之前，此时默认参数调整不需要带名称，因为排在前面优先了。
    print(x)
    print(a)
    print(args)
    print(kwargs)
foo(1,2,3,4,5,6)   # 此时即使不写a,由于默认参数在形参定义在前面优先级高，能先吃到一个数，剩下的再全部给位置参数吃掉。

def foo(x,a=4, *, b, **kwargs):  # 此时*只是作为位置参数与关键字参数的分隔符，不代表任何args,也就指定了b为关键字参数
    print(x)
    print(a)
    print(b)
    print(kwargs)
foo(1,2,b=6)


# 是否有必要区分
def foo(*, x, **kwargs):
    print(x)
    print(kwargs)
foo(x=2,z=1)       # 形参能识别出来是关键字参数，则必需用等号


# 默认都会认为是位置参数，除非通过*之后强制或通过**kwargs强制定义为关键字参数
data1 = dict(img=1, img_meta=2, gt_label=4)
def foo1(img, img_meta, loss=True,**kwargs):  # 默认都是位置参数或默认参数
    print(img)
    print(img_meta)
    print(loss)
    print(kwargs)
foo1(**data1)                        # 位置参数或默认参数的实参，带不带等号传入都可以

data2 = dict(img=1, img_meta=2, gt_label=4)
def foo2(img, img_meta, loss=True):  
    print(img)
    print(img_meta)
    print(loss)
foo2(**data2)                        # 解包数据需要确保形参都能接收，多余的数据需要通过**kwargs接收，否则会报错


# 

"""-------------------------------------------------------------------------
Q. 如何理解*args和**kwargs的真正应用区别？
关键理解：
*args可以输入一批数据value，并会被自动转化成列表
**kwargs可以输入一批数据key=value，并会被自动转化成字典
"""
# args的功能是自动把输入参数变成了一个可迭代的list
def args_test(param1,*args):
    print("first param is:",param1)
    index = 2
    for value in args:
       print("the "+str(index)+" is:"+str(value))
       index += 1
       
data = (7,5,9,2)
args_test(7,5,9,2)  # 2种参数传递方式等效
args_test(*data)    # 2种参数传递方式等效， 这种更常用，调用代码更简洁
args_test(2, a=1,b=2)

# kwargs的功能是自动把输入参数变成了一个可迭代的dict
def kwargs_test(param1,**kwargs):
    print("the first param is: ",param1)
    for key in kwargs:
        print("the key is: %s, and the value is: %s" %(key,kwargs[key]))
        
data = dict(a=5,b=3)
kwargs_test(2, a=5, b=3)  # 2种参数传递方式等效
kwargs_test(2, **data)    # 2种参数传递方式等效， 这种更常用，调用代码更简洁




'''-------------------------------------------------------------------------
Q: 函数返回的多个变量是怎么存在的，怎么获得？
核心概念：
1. 函数的输入参数：必须是解包的独立参数
2. 函数的输出参数：通过return输出，如果写成return a,b,c，系统默认会把输出组合成tuple (a,b,c)
   如果写成return (a,b,c)，系统也不会再进行重复打包成双层tuple
'''
def myfun(x,y,z):
    return x+1, y+2, z+3  # 注意，传入的参数需要解包传入，即散装传入，但传出是默认打包传出！
input = (1,2,3)           # 传入前：turple
output = myfun(*input)    # 解包后传入，传出自动打包为turple, 即使自己写成return (a,b,c)也不会重复打包成双层tuple
x,y,z = output           # 解包操作赋值给多个变量
print(x,y,z)

i,j,k = myfun(*input)   #也可以输出直接写成解包形式
print(i,j,k)


'''-------------------------------------------------------------------------
Q. 函数的continue/break/的区别？
- break用于跳出所在循环层到外层
- continue用于跳出所在单次循环
'''
for i in range(5):
    print('loop {}---------'.format(i))
    for j in range(10):
        if j == 3:
            break  # 只在j=3时终止整轮打印，内层循环停止，下一轮外层继续
        print(j)

for i in range(5):
    print('loop {}---------'.format(i))
    for j in range(10):
        if j == 3:
            continue  # 在j=3时终止该次打印，内层循环未停
        print(j)


'''-------------------------------------------------------------------------
Q: 对于一些简单的功能，如果不想编正经函数，有没有简洁方式，比如匿名函数？
'''
add = lambda x, y: x + y  # 可以理解为lambda就是一个匿名函数名，后边跟变量名，冒号后跟函数体
add(2,3)


'''-------------------------------------------------------------------------
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


# 高阶函数的主题参考：
# https://www.cnblogs.com/cwp-bg/p/8859260.html
'''-------------------------------------------------------------
Q. 高阶函数 - map怎么用？
iterator = map(func, iterable)
map的功能是把迭代对象中的元素依次送入函数中，返回一个迭代器(next方法)(python2.x返回list)
'''
def ff(x):
    return x**2
a = map(ff, [1,2,3])
for i in a:
    print(i)
print(list(map(ff, [1,2,3])))


'''-------------------------------------------------------------
Q. 高阶函数 - reduce怎么用？
'''



'''-------------------------------------------------------------
Q. 高阶函数 - filter怎么用？
'''



'''-------------------------------------------------------------
Q. 高阶函数 - partial怎么用？
1. 核心理解：partial可以理解为一个wrapper, 用于事先把部分参数跟原函数封装起来，调用起来就可以更少输入参数，更简洁
   作为wrapper，也就可以使用update_wrapper(wrapper, src_func)做属性更新
'''
from functools import partial, update_wrapper
def func(name, age):
    """this is func __doc__"""
    print('my name is {}, my age is {}'.format(name, age))

wrapper = partial(func, age = 36)
update_wrapper(wrapper, func)

wrapper('eason')
print(wrapper.__doc__)


'''-------------------------------------------------------------
Q. 高阶函数 - sorted/max/min怎么用？
'''


'''-------------------------------------------------------------
Q. bisect库里边的二分法排序如何实现
1. 核心理解：2大类命令，bisect用于查找，insort用于插入
    bisect.bisect(list, data)查找如果插入对应index是多少，默认在相同元素右侧
    bisect.bisect_right(list, data)在相同元素右侧
    bisect.bisect_left(list, data)在相同元素左侧
    bisect.insort(list, data)插入数据，默认在相同元素右侧
    bisect.insort_right(list, data)在相同元素右侧插入输入
    bisect.insort_left(list, data)在相同元素左侧插入输入
2. 查找
'''
import bisect
import random
random.seed(1)
print('---  --- --------')
l = []
for i in range(1, 15):
    r = random.randint(1, 100)   # 随机生成一个数
    position = bisect.bisect(l, r)  # 在数组l中查找数字r的index返回，但不执行插入操作
    bisect.insort(l, r)             # 在数组l中插入r
    print('%3d  %3d' % (r, position), l)


l1 = [1,2,3,4,5,6,7]
p = bisect.bisect_right(l1, 4)  # 查找
bisect.insort(l1, 4)    # 插入，如果存在相同数，则放在左侧
print(l1)

# 二分排序法：用bisect查找比循环算法以及递归算法都要更快
# 参考：http://python.jobbole.com/86609/
def binary_search_loop(lst,x):  
    """普通循环二分法查找"""
    low, high = 0, len(lst)-1  
    while low <= high:  
        mid = (low + high) / 2  
        if lst[mid] < x:  
            low = mid + 1  
        elif lst[mid] > x:  
            high = mid - 1
        else:
            return mid  
    return None

def binary_search_bisect(lst, x):
    """bisect查找"""
    from bisect import bisect_left
    i = bisect_left(lst, x)
    if i != len(lst) and lst[i] == x:
        return i
    return None

# 对比2种内置命令查找index的方法：
import numpy as np
lst = [1,4,6,7,10,13,15,18,21,32,42,55]
print(np.where(np.array(lst)==13))

print(bisect.bisect_left(lst, 13))



'''-------------------------------------------------------------
Q. 什么叫函数的闭包，有什么作用？
1. 全局变量(在模块/函数/class外部的，叫做全局变量)
2. 闭包：就是记住了嵌套作用域变量值的函数，形成一个闭合的存储包，他有点像一个类的实例，有自己的属性值和方法。
   而闭包更简单只是一个带部分参数值的函数对象。
3. 

参考：https://segmentfault.com/a/1190000007321972
'''

# 理解闭包先要理解python中如何划分局部变量和全局变量
num = 100
def func():
    num += 50      # 创建或修改num变量时：需要先赋值，这里没有赋值过所以报错
    print(num)
func()
# 微调后的代码：
num = 100
def func():
    x = num +50    # 使用且不修改num变量时：先局部找，找不到再全局找
    print(x)  
# 二次微调后的代码：
num = 100
def func():
    num = 50
    x = num +20    # 相同变量名如果既有全局的也有局部的，则优先取局部变量
    print(x)
# 三次微调后的代码：
num = 100
def func():
    global num
    num += 50     # 在局部函数想要修改全局变量，需要采用global声明
    print(num)
func()

# ------------闭包------------
# 普通闭包：通常函数作用域内的变量在函数结束后自动小时，但闭包能够把函数作用域中的变量永久保留
# 所以闭包就是记住嵌套作用域变量值的函数，形成一个闭合的存储包。主要是指内层函数为闭包，在函数运行结束后，依然存在存储包
def func():
    x = 30
    def wrapper():
        return x +100   # 局部变量没找到，从外层找，找到后放入闭包持久保存
    return wrapper

func()()               # 执行函数func()返回的是一个函数对象，然后要执行返回的函数对象就要再加一对括号
func().__closure__     # 闭包函数名会多一个__closure__属性，该属性是一个元组，每个元组元素都是一个cell对象放置外部变量
func().__closure__[0].cell_contents

# 普通装饰器，也是一种闭包
def deco(f):
    def wrapper(*args, **kwargs):
        print('this is wrapper')
        f(*args, **kwargs)          # 从外层找到变量对象f作为局部变量使用
    return wrapper                  # 这个wrapper函数对象就已经是一个闭包，里边包含了传入的一个函数对象
@deco
def ori():
    print('this is ori')
ori()  
ori.__closure__[0].cell_contents   

# 带参装饰器，更是一种闭包
def func(a,b):
    def deco(f):
        c = 10
        def wrapper(*args, **kwargs):
            f(*args, **kwargs)
            print('my age is {}'.format(c+a+b))
        return wrapper
    return deco
@func(1,2)    # 等效于ori = func(1,2)= deco = 
def ori():
    print('this is ori')
ori()
ori.__closure__[0].cell_contents
ori.__closure__[1].cell_contents
ori.__closure__[2].cell_contents
ori.__closure__[3].cell_contents

