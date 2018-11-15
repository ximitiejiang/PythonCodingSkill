#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:27:22 2018

@author: suliang
"""

# 迭代器和生成器

'''
Q: 想要迭代一个序列，但想同时记录下迭代序列当前的元素索引，怎么做？
'''
mylist = ['a', 'b', 'c']
for id, value in enumerate(mylist): # 使用内置enumerate()函数同时获得序列的索引和数值
    print(id, value)


'''
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

    
    
'''
Q: 想要同时迭代多个序列，怎么做？
'''
listx = [1,5,4,2,10,7]
listy = [101,78,37,15,62,99]
# 使用内置的zip(a, b)函数构建一个迭代器元祖(a, b)
# zip()也可以接受2个以上的序列组成迭代器zip(a, b, c)
for x, y in zip(listx, listy):  
    print(x, y)
    

'''
Q. 使用yield指令返回的迭代器有什么用？
'''
def fab(max): 
    '''生成一个斐波那切数列：一次性生成整个存在内存里，供调用
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
#--------------------------------------------------
def fab(max):
    '''生成一个斐波那切数列：加入yield b等效于返回b
    因为没有return返回值了，此时fab()不再是普通函数，而是迭代器了，
    需要哪个数据时，函数马上生成，所以不占很多内存，每次只有一个数据占内存。
    迭代器没有__len__()方法，也不能打印，类似range(5)的使用。但有内部的__next__()方法，
    由于不占内存，已广泛用在nn.parameters这类函数输出上了。
    '''
    n, a, b = 0, 0, 1 
    while n < max: 
        yield b 
        a, b = b, a + b 
        n = n + 1 

for i in fab(5):      # 迭代器仅有属性：赋值给循环变量
    print(i)

    
'''
Q: 想要执行一类事先要设置，事后要清理的工作，比如打开文件后要关闭，比如？
'''
# with语句先执行with之后的语句，如果为真，则调用__enter__方法返回值赋值给as后的变量
# with执行完后，调用__exit__方法
with open('xxx.txt', 'wt') as f:  # 写入模式打开文件
    f.write(text1)               # 写入文件    
    
    
    
    
    
    