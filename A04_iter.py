#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:27:22 2018

@author: suliang
"""

# 迭代器和生成器

'''---------------------------------------------------------------------------
Q: 为什么要一个迭代器，现实中有哪些迭代器，怎么创建一个迭代器？

1. 迭代器目的：生成一个循环数. 跟常规数组之类比起来，迭代器不需要先生成并加载内存
    而是需要一个数生成一个数，
    只要实现了__next__方法的就是一个迭代器(python2是用next()函数)
    在内部，而for循环就是调用__next__方法

2. 核心概念的区别：
    (1)迭代对象iterable：
        拥有__iter__()函数，返回的就是迭代对象
        方法1：for循环可用(for循环内部把它转化为迭代器)
        方法2：iter()函数把迭代对象转化为迭代器       （重要）
    (2)迭代器iterator：
        拥有__iter__()函数返回的迭代对象，也拥有__next__()函数返回每个元素
        方法1：for循环可用
        方法2：next()函数可用，获得一个单独的元素     （重要）
        方法3：list()函数可用，获得所有元素           （重要）
    (3)生成器generator
        更便捷的迭代器，同样节省内存，且不需要手动去实现__iter__()和__next__()
        方式1：圆括号的列表推导式(i**2 for i in range(10000))
        方式2：带yield返回的函数

3. 常用迭代对象/迭代器工具函数
    (1)迭代对象：可以直接用于for循环
    list/dict/set/tuple
    range()
    
    (2)迭代器
    enumerate()
    zip(lst1, lst2)
    map(func, list)  
    iter()    正向迭代器
    reverse() 逆向迭代器
        
'''
# --------迭代对象---------------
for i in range(5):  # 用range()生成一个迭代对象，但不是迭代器，所以不能用next()函数
    print(i)

lst = [2,4,6,8]     # 所有list/dict/tuple/set都是迭代对象，但不是迭代器，所以不能用next()函数
for i in lst:
    print(i)

# --------迭代器---------------
data = ['a','b','c']
for i, value in enumerate(data):  # 用enumerate()函数生成一个迭代器,返回元组，然后元组解包
    print(i, value)

xpts = [1,3,5,7]
ypts = [100,300,500,700]
for i in zip(xpts, ypts):     # 用zip()函数生成一个迭代器，返回元组，可以解包，也可直接输出元组
    print(i)

lst = [1,3,8,15,21]
for i in iter(lst):      # 用iter()函数创建一个正向迭代器
    print(i)

for j in reversed(lst):  # 用reversed()函数创建一个反向迭代器
    print(j)

# --------迭代器的访问，除了for循环，还可以next()---------------
data = ['a','b','c']
next(enumerate(data))

# --------创建一个迭代器---------------
class Fib(object): 
    def __init__(self, max): 
        super(Fib, self).__init__() 
        self.max = max 
        self.a = 0 
        self.b = 1 
    def __iter__(self):     # 定义__iter__从而是一个可迭代对象，且return自己
        return self 
    def __next__(self):     # 定义__next__从而是一个迭代器，返回next的值，每次计算一个元素，而不用一次性把所有元素计算好存在内存
        fib = self.a 
        if fib > self.max: 
            raise StopIteration 
        self.a, self.b = self.b, self.a + self.b         
        return fib
    
fib = Fib(5)
for i in fib:
    print(i)


'''---------------------------------------------------------------------------
Q: 如何在迭代器基础上改成生成器，生成器比迭代器有什么优势？

1. 生成器本质也是迭代器，所以能用for循环和next()函数
   但生成器创建更简单，不需要自己实现__iter__和__next__方法
   
2. 生成器创建方法1: 列表推导式，把[]换成()，适合生成逻辑简单的数据生成器
   生成器创建方法2: yield改造函数，适合生成逻辑复杂的数据生成器
'''
# --------创建简单生成器---------------
g1 = (i**2 for i in range(5))

for i in g1:
    print(i)

next(g1)
# --------创建复杂生成器---------------
def G2(max):
    for i in range(max):
        if i%5 == 0:
            yield i

g2 = G2(30)
for i in g2:
    print(i)


'''---------------------------------------------------------------------------
Q: 想要迭代一个序列，但想同时记录下迭代序列当前的元素索引，怎么做？
'''
mylist = ['a', 'b', 'c']
for id, value in enumerate(mylist): # 使用内置enumerate()函数同时获得序列的索引和数值
    print(id, value)


'''---------------------------------------------------------------------------
Q: 想要迭代一个字典，但想同时迭代字典的键和值，怎么做？
'''
# 同时循环键 + 值： for key, value in dic.items()
# 只循环键： for key in dic.keys()
# 只循环值： for value in dic.values()
mydict = {'alen':3, 'backer':2, 'cathy':4,'leo':9, 'eason':20}
for key, value in mydict.items():# 使用.items()方法同时获得键和值
    print(key, value)

for key in mydict.keys():
    print(key)
    
for value in mydict.values():
    print(value)

    
'''---------------------------------------------------------------------------
Q: 想要同时迭代多个序列，怎么做？
'''
listx = [1,5,4,2,10,7]
listy = [101,78,37,15,62,99]
# 使用内置的zip(a, b)函数构建一个迭代器元祖(a, b)
# zip()也可以接受2个以上的序列组成迭代器zip(a, b, c)
for x, y in zip(listx, listy):  
    print(x, y)
    

'''---------------------------------------------------------------------------
Q. 迭代对象，迭代器，生成器，实现方案时有什么区别？
'''
def fab(max): 
    '''生成一个斐波那切数列：采用的是迭代对象（但不是迭代器），也就是用容器来预装好数据
    一次性生成整个存在内存里，供调用
    缺点是所有数据都放在内存，比较占内存
    '''
   n, a, b = 0, 0, 1 
   L = [] 
   while n < max: 
       L.append(b) 
       a, b = b, a + b 
       n = n + 1 
   return L

print(fab(5))

#--------用迭代器方案--------------
class Fab():
    '''生成斐波拉切数列：采用迭代器方案
    需要next数据时，就计算生成一次，不会把所有数据都计算出来放内存
    优点是比较省内存空间
    '''
    def __init__(self,max):
        self.a=0
        self.b=1
        self.max=max
    def __iter__(self):
        return self
    def __next__(self):
        fab = self.a
        self.a, self.b = self.b, self.a + self.b
        return fab
fab = Fab(10)
next(fab)

#--------用生成器方案--------------
def fab(max):
    '''生成一个斐波那切数列：加入yield b等效于返回b
    因为没有return返回值了，此时fab()不再是普通函数，而是迭代器了，
    需要哪个数据时，函数马上生成，所以不占很多内存，每次只有一个数据占内存。
    迭代器需要手动实现__iter__和__next__，而生成器yield会自动生成这两个函数而变为迭代器。
    可见生成器结构更简单，可以在一个函数基础上修改，实现next的功能即可。
    '''
    a, b = 0, 1
    n = a
    while n < max: 
        yield a 
        a, b = b, a + b 
        n = a 

for i in fab(10):      # 迭代器仅有属性：赋值给循环变量
    print(i)

    
'''---------------------------------------------------------------------------
Q: 想要执行一类事先要设置，事后要清理的工作，比如打开文件后要关闭，比如？
'''
# with语句先执行with之后的语句，如果为真，则调用__enter__方法返回值赋值给as后的变量
# with执行完后，调用__exit__方法
with open('xxx.txt', 'wt') as f:  # 写入模式打开文件
    f.write(text1)               # 写入文件    
    
    
lst = [14,6,9]
next(lst)    
    


'''---------------------------------------------------------------------------
Q: 如何处理map,zip函数返回的迭代对象
- iter_obj3 = zip(iter_obj1, iter_obj2): 将一组iterable对象打包压缩，返回的是一个iterable对象(即多个对象变一个对象)，每个元素是一个元组
- *(zip_obj)则解压缩一个对象为多个对象。
- iter_obj = map(func, iter_obj): 将func作用于一个iter对象的每一个元素，然后返回一个iter对象
- 

'''
# zip函数----------------------------
import numpy as np
a=[1,2,3]
b=[4,5,6]
c=[7,8,9]
zz=zip(a,b,c)
print(list(zz))  # list()函数解放迭代对象
x,y,z=zip(*zz)   # *操作解放压缩对象
print(x)
print(y)
print(z)

# map函数-----------------------------
def f1(x):
    return x**2
def f2(x,y):
    return x+y
a1 = [1,2,3]
a2 = [4,5,6]
b = map(f1, a1)   # map() 传入1个函数和1个迭代对象参数
c = map(f2, a1, a2) # map() 传入1个函数和2个迭代对象参数
print(list(b))
print(list(c))
print(*b)
print(*c)

def foo(*args, **kwargs):
    print(args)
    print(*args)
a = [1,2,3]
print(foo(a))
