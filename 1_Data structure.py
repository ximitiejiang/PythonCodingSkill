#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:23:37 2018

@author: suliang
"""
# List, Dictionary

'''
Q: 如何深度理解list的append?
'''
# 新建一个空的lst=[]之后，由于没有编号，任何lst[i]的操作都是报错的，因为无法定位到准确内存
# 任何切片只能获取已经存在的内存地址不能凭空创建，所以只能通过lst.append(value)来从最后一个位置写入数据。
# lst.append(value)是一种浅复制(类似copy, d[:])，如果value是list，而list内部变化会引起lst里边的值的改变
lst = []
lst.append([1,2]) 
# lst[1] = 200 左边这是错误写法， 因为： 空list要配append, 初始化全0的list就可以配切片
a0 = [20]
lst.append(a0)
print(lst)
a0.append(4)  # 如果a0内部变化，会导致lst变化，因为append是一种浅复制。
print(lst)

# 同时需要对比跟append相关的2个函数extend和insert
lst = [1,2,3,4,5,6]
lst.append([1,2])   # 在末尾添加一个元元素
lst.extend([1,2])   # 在末尾添加n个元素
lst.insert(-1,[1,2])  # 在index位置添加一个元素

'''
Q: 如何理解切片的写法？

从0开始并且不考虑最后一个元素的方式有如下优点：
1. lst[:3]就可以直接读出包含了3个元素，这跟range(3),np.arange(3)有相同功效
2. lst[:3]和lst[3:]正好把list分成两部分，非常直观
3. lst[start:stop]可以通过起点和终点直接算出包含元素个数n=stop-start
'''
lst = [3,4,2,6,1,6,4]
print(lst[:3])   # 写成冒号3，正好代表有3个元素，非常自然
print(lst[3:-1]) # 写成3到最后，正好代表下一个元素到最后，从而把3当成分界点
# 把3当成分界点，冒号3代表包含3个元素，3冒号代表3个元素之后（参考流畅的python）

# 核心要记住a[a:b:c]中a代表起点，b代表终点，c代表间隔
# 以下是几个变种
a = [[1,2,3,4,5,6,7,8]]
a[1:8:2]
a[:8:2]
a[1:8]
a[1:-1:2]
a[:, 1::4]  


'''
Q: list里边嵌套list是算法里边常见的一种数据结构，如果对这类list进行切片？
'''
lst = [[1,2,3],[4,5,6],[7,8,9]]
for i in range(len(lst[0])):
    featList = [sample[i] for sample in lst]  # 循环是取得list列的标准写法


'''
Q: 创建dict字典最重要的2种方法?
'''
# 这是最符合应用习惯的dict生成方法: 广泛用于函数参数列表的输入
params = dict(type='SGD', lr=0.01, weight_decay=0.1)
# 这是最符合应用习惯的dict添加元素方法
params.update(decay = 0.02)    # update方法类似与list的extend方法

# 这是普通dict生成方法
age = {'leo':18, 'eason':8, 'winnie':16}


'''
Q: 有一个包含N个不同类型元素的序列(list/dic/str)，如何把它分解为N个单独的变量？
'''
p = [12,5]
q = 'hello'
x,y = p           # 用多变量方式可以分解list成多个变量
a,b,c,d,e = q    # 用多变量方式可以分解string成多个变量

data = ['ACME', 50, 91.1, (2012, 12, 2)]
name, shares, price, (year, mon, day) = data  # 用多变量嵌套格式可以分解出多个变量
name, _, price, _ = data   # 用下划线方法可以代表某个待丢弃掉的变量

record = {'Dave','dave@example.com','773-555-1212','847-555-1212'}
name, email, *phone = record  # 用*name变量的形式可以表示开头/中间/末尾多个数据集

record = ['ACME', 50, 91.1, (2012, 12, 2)]
name, *_, (year, *_) = record  # 用*_代表多个待丢弃的变量


'''
Q: 有一个字典，怎么对字典里边的值进行计算(最大值，最小值，排序)？
'''
# 使用z = sorted(zip(a, b))， 可以对字典，tuple等各种组合数据进行排序，非常有用
price = {'ACME': 45.23,
         'AAPL': 612.78,
         'IBM': 205.55,
         'HPQ': 37.2,
         'FB': 10.75}
max_price = max(zip(price.values(),price.keys()))  # zip()结合max()求最大最小值
sorted_price = sorted(zip(price.values(),price.keys())) # zip()结合sorted()排序
# zip()创建了一个迭代器，并把数值放前面，便于使用各种函数比如sorted()
# 该方法优于 max(price.values())，因为能同时返回key和value
# 如果遇到2个最大/小值情况，返回的是两个中键更大/小的那个


'''
Q: 有一组数据，如何获得最大的或最小的N个元素？
'''
num = [1,8,2,23,7,-4,18,23,42,37,2]
max_3 = num.sorted()[:3]   # 通过先排序再切片的方式获得最大/最小的N个元素
# 这种写法非常有意思，可以方便获得从小到大排序和从大到小排序
min_to_max = num.sorted()[::]
max_to_min = num.sorted()[::-1]

min_to_max_index = num.argsort()[::]
max_to_min_index = num.argsort()[::-1]

'''
Q: 想创建一个字典，并且希望字典内保持输入时的顺序，而不是默认的key字母顺序？
'''
from collections import OrderedDict
d = OrderedDict()   # 通过OrderedDict类创建一个带顺序的字典，按添加顺序保存
d['foo'] = 2
d['bar'] = 1
d['spam'] = 3
for key in d:
    print(key, d[key])


'''
Q: 有一组序列，希望找到序列中出现次数的元素，甚至每个元素的出现次数，怎么做？
'''
lst1 = [1,3,2,4,5,7,8,9,1,2,6,9,0,
       8,3,1,8,5,2,8,3,0,3,6,8,9,
       0,9,7,4,7,3,1,7,5,7,0,4,0]
lst2 = [0,0,1,0]
from collections import Counter
num_count = Counter(lst1)   # 使用Counter()函数，可以统计每个元素出现次数，返回list或dict
num_count.update(lst2)      # counter.update()可以增加计数变量的update方法
num_count[0]                # 类似字典或list的调用方法: 默认是按照key排序而不是value


'''
Q: 有一组序列，希望对某个键进行排序，用sorted（）怎么定义？
'''
rows = [{'fname':'David', 'lname':'Jones', 'uid': 1003},
        {'fname':'Brian', 'lname':'Beazley', 'uid': 1002}]
from operator import itemgetter
sort_fname = sorted(rows, key = itemgetter('fname')) 
# 用itemgetter()函数获得item的‘fname’字段

data = [('b', 2), ('a',3), ('c',1)]
sort_data = sorted(data, key = itemgetter(0))  
# 用itemgetter()获得每个item的第0元素或第1元素


'''
Q: 如何排序，有哪些不一样的排序方法？
'''
lst = [2,1,4,2,7]
s1 = lst.sort()   # 这是对lst的永久排序

lst2 = [2,1,4,2,7]
s2 = sorted(lst2)  # 这是临时排序，不影响lst2
# sorted.(lst)放在list前面，lst.sort()放在list后面
# sort函数可以对任何iterable的结构进行排序：
# 比如dict, list, str, dic.value, dic.key, tuple

arr = [('a', 1), ('b', 2), ('c', 6), ('d', 4), ('e', 3)]
s3 = sorted(arr, key=lambda x:x[1])
# sorted还可以带一个函数，相当与先对x进行函数处理，然后排序，但输出结果是原数据而不是处理过的x


'''
Q: 有一组序列，我知道怎么排序，但不知道怎么筛选出特定要求的元素？
'''
import numpy as np
mylist = [1,4,-5,10,-7,0,3,-1]
pos = [n for n in mylist if n > 0]     # 用列表推导式做筛选-格式1(先循环后判断)
pos_or_zero = [n if n>0 else 0 for n in mylist] # 用列表推导式筛选-格式2(先判断后循环)

index_gtz = np.where(np.array(mylist)>=0)  # 用numpy的where()函数可以返回index
index_ltz = np.where(np.array(mylist)<0)

# 列表推导式 [n for xx if xxx] 先for循环，再条件判断
# 列表推导式 [n if xx for xxx] 先条件判断，再进行for循环


'''
Q: 一组序列的筛选可以通过大于小于，那两组序列的交集并集差集如何筛选出来？
'''
a_list = [1,2,3,4]
b_list = [1,4,5]

ret_list = [item for item in a_list if item not in b_list]  # a对b的差集
ret_list = [item for item in a_list if item in b_list]  # 交集

# 高级写法求差集，并集，交集
ret_list = list(set(a_list)^set(b_list)) # 差集: 用 ^ 
ret_list = list(set(a_list).union(set(b_list))) # 并集： 用a.union(b)
ret_list = list((set(a_list).union(set(b_list)))^(set(a_list)^set(b_list))) # 交集

# 高级写法2



'''
Q: 为什么有时候复制一个变量会影响另一个变量，有时候又不会影响？
'''
# 这是一个重要知识点：在算法编写过程中，已经踩坑2次，每次耗费我2天都找不到问题根源
''' 1. python中一类叫不可变对象，包括数字/字符串/元祖，他们的复制只能创建新内存，所以是不会影响
    2. python中另一类叫可变对象，包括数组/字典，他们常规复制都是浅复制，不同变量名指向同一内存地址
    3. 相应的在函数形参上，不可变对象形参是传值，函数体内改变该对象，不会影响函数体外的原始变量；
       而可变对象形参是传址，即传递的是指针，在函数体内改变该对象，会影响函数体外的原始变量。
    4. 对于b=a[:], b=list(a), b=a*1, b=copy.copy(a)四种方式复制列表，如果列表没有嵌套，这几种方式相当与取到值，所以算深拷贝，
       结果都可以得到一个新的列表
       
       最可靠的深复制方式：import copy; copy.deepcopy(lst); 这时唯一一种深复制，无论对象是否有嵌套
'''

old = [4,1,3,['age',10]]
new_1 = old
new_1.append(100)
print(old, new_1)   # 用名称的形式拷贝，只是指针拷贝，指向同一内存，共同变化

old = [4,1,3,['age',10]]
new_2 = old[:]      # 用old[:]的形式是浅拷贝，对外层的所有对象都可以拷贝了
new_2.append(100)
print(old, new_2)
id(old),id(new_2)

old = [4,1,3,['age',10]]
new_2 = old.copy()      # 用copy()的形式是浅拷贝，对外层的所有对象都可以拷贝了
new_2.append(100)
print(old, new_2)

old = [4,1,3,['age',10]]
new_2 = old.copy()      # 用copy()的形式是浅拷贝，在修改内层可变对象时，依然会影响源数据
new_2[3].append(100)
print(old, new_2)


old = [4,1,3,['age',10]]
import copy
new_2 = copy.deepcopy(old)  # 要做到完全深度拷贝，不影响源数据，唯一方法是deepcopy()
new_2[3].append(100)
print(old, new_2)

# 另一种深复制：直接取到元素，元素作为不可变对象，肯定是创建新内存
old = [4,1,3,['age',10]]
new3 = [old[0],old[1],old[2],[old[3][0],old[3][1]]] # 此处如果取old[3]也不行，因为old[3]是list而不是不可变对象
old[3][1]=100
new3  # 可以看到虽然old[3][1]变了，但new1没有受影响



'''
Q: 如何生成正态分布的随机数据？
'''
import numpy as np
'''正态分布
'''
r1 = np.random.randn(2,4)   # 生成2x4的标准正态分布N(0,1)
r2 = 5*np.random.randn(2,4)+10  # 生成2x4的正态分布N(10,5)
'''均匀分布 - 实数
'''
r3 = np.random.rand(2,4)    # 生成2x4的均匀分布
'''均匀分布 - 整数
'''
r4 = np.random.randint(5, size=(2,4))  # 生成[0,5]之间整数的均匀分布
r5 = np.random.random_integers(5, size=(2,4)) # 生成[1,5]之间整数的均匀分布


'''
Q: 如何把只含一个元素的array或者list转化为标量的float/int？
'''
# 先用type()函数查看数据类型，到底是矢量nunpy/list，还是标量float/int
type(x)  


'''
Q: 如何判断None这种变量？
方式1：用is/is not判断
方式2：用x/not x判断
'''
# 布尔变量的两种写法：方法1(is/is not语句做判断)，方法2(直接作为判断句)
x = False
if x is False: print('x is False')
if x: print('x is True')

# None的判断可以用方法1： is/is not
x = None
if x is None: print('x is None')  # is None
if x is not None: print('x is not None')
# None的判断也可以用方法2，此时None/0/空列表[]/空元组()/空字典{}都可以等效于False
if not x: print('x is None')
# 最佳实践：尽量用not x这种写法，而不要用==或者is，也就是尽量用方法2(直接判断 if.., if not..)






