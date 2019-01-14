#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:23:37 2018

@author: suliang
"""
"""单句总结----------------------------------------------------------------
1. list/tuple的+/*是重载成串联，但对元素的+/*需要通过列表推导式完成，优化的array的元素操作可以用+/*重载完成
2. 筛选数据：相当于元素级操作，list用列表推导式+条件判断，优化的array可用>/<重载完成
3. 查看类型type(), 优化的array查看修改数据类型d.dtype, d.astype(np.int32)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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
Q. list/tuple的相加和相乘的效果？
- __add__重载运算符
'''

# list/tuple/str的相加: 原理一样，进行拼接
l1 = [1,2,3]
l2 = [4,5,6,7,8]
l3 = l1 + l2     # 只有同种类型数据能相加：相加相当于首位串联，这是加号重载运算符的结果

str1 = 'Hello'
str2 = 'World'
print(str1+str2)  # 加法重载，很适用于string的串联

d1 = (4,5,6)
d2 = [1,2,3]
d3 = dict(a=1,b=2)
d4 = d1 + (d2, d3)  # 不同类型的数据如果要相加，需要先把不同类数据组合成同类数据

# list/tuple的相乘
b1 = [1,2,3]
b2 = b1 * 3        # 相乘就是多个相加

# 那么按位操作元素进行相加/相乘怎么做？
l1 = [1,2,3]
l1 + 1 # list不能跟标量相加，报错
l1 * 2 # list乘标量，是运算符重载的叠加，不是按位操作
[n+1 for n in l1]  # 通过列表推导式对元素操作
[n*2 for n in l1]  # 通过列表推导式对元素操作

# numpy的优势：可以直接运算符操作元素(+/*/</>)
arr1 = np.array([1,2,3])  # 通过转成ndarray后就可直接元素操作
arr1 + 1      # ndarray的数组可以跟标量直接相加（想当于对每个元素操作）
arr1*2        # ndarray的数组可以跟标量直接相乘（想当于对每个元素操作）
arr1 > 2      # ndarray的数组可以跟标量直接比大小，返回bool列表


'''
Q: 如何理解切片的写法？

从0开始并且不考虑最后一个元素的方式有如下优点：
1. lst[:3]就可以直接读出包含了3个元素，这跟range(3),np.arange(3)有相同功效，这个可以叫做去尾效应
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
Q. 更高级的切片写法
a[m,:] 代表提取第m行
a[[m,n],:] 除了代表提取第m，n行，还需要按照m,n的顺序，相当于调整行顺序
'''
from numpy import random
random.seed(11)
a = random.randint(0,10, size=(3,3))

a[(1,2,0),:]
a[[1,2,0],:]

flag = 3
a[0,:]     # 提取第0行
a[(1,1,0),:]  # 连续提取第1行2次和第0行
'''
Q: list里边嵌套list是算法里边常见的一种数据结构，如果对这类list进行切片？
'''
lst = [[1,2,3],[4,5,6],[7,8,9]]
for i in range(len(lst[0])):
    featList = [sample[i] for sample in lst]  # 循环是取得list列的标准写法


'''------------------------------------------------------------------------
Q: 创建dict字典最重要的2种方法?
'''
# 这是最符合应用习惯的dict生成方法: 广泛用于函数参数列表的输入
params = dict(type='SGD', lr=0.01, weight_decay=0.1)
# 这是最符合应用习惯的dict添加元素方法
params.update(decay = 0.02)    # update方法类似与list的extend方法

# 这是普通dict生成方法
age = {'leo':18, 'eason':8, 'winnie':16}



'''------------------------------------------------------------------------
Q: dict字典最常用的方法?
'''
d1 = dict(a=1,b=2,c=3,d=4)
d1['e'] = 5  # 增加key/value
d1['b'] = 9  # 修改value

d1.update(h=5)  # 增加key/value，相对来说dict(a=1), d.update(a=1)这两个命令都是dict常用，因为比较书写简单。

d1.get('a')
d1['f']      # 通过key切片获得value，没有报错
d1.get('f')    # 获得某key的value，用get不会报错，没有就返回None。可用于判断是否None，不必终止程序

'''------------------------------------------------------------------------
Q.怎么使用OrderedDict和defaultDict?
'''
# defaultdict
from collections import defaultdict, OrderedDict
d1 = defaultdict(int)
name = 'mississippi'
for k in name:
    d1[k] += 1        # 使用defaultdict在存储不存在键的值时不会报警
                      # 但使用其他dict就会报警
# OrderedDict
d2 = OrderedDict(m=2,a=1,c=3)
d3 = dict(m=2,a=1,c=3)

d4 = OrderedDict([('m',1),('a',2),('c',3)])

'''------------------------------------------------------------------------
Q.在数据这一块，有多少类型，怎么转换?、
1.如果不知道是什么结构，就用type(var)查看，号称万用的type()
2. 如果知道是数据，就可以用arr.dtype / arr.astype() 这2兄弟查看加转换
    >包括np.float64, np.int32, ...
'''
d1 = np.array([1,2,3])   
d1.dtype                    # dtype查看： 后置不带括号，list不适用
d2 = d1.astype(np.float32)  # 先转array，然后用astype()

# 对python中对象类型的判断，可以用type(),但更推荐用isinstance(obj, (type_tuple))
# 因为实际使用中type()的逻辑不太符合需求:
# type()区分较严格，把子类对象和父类认为是不同类型，但大多数应用情况我们希望是同类
# isinstance()通常把子类对象也归属于父类
class A:
    pass
class B(A):
    pass
type(A())==A   # 对象永远属于类  (对象是类的汗毛变的，永远属于类)
type(B) == A   # 继承类不属于父类  (继承类是类的儿子，但实际是2家人)
type(B())==A   # False,继承对象的类不属于父类 (type严格区分)

isinstance(A(),A) # 对象永远属于类
isinstance(B,A)   # 继承类不属于父类
isinstance(B(),A)  # True,继承对象属于父类  (isinstance符合实际，子类的汗毛也属于父类)
# 具体类别
# python: int, float, str, bool, list, dict, tuple
# numpy: np.int64, np.float64

a1 = 0.2         # float
a2 = [1,2]       # list
a3 = dict(a=1)   # dict
a4 = 'hello'     # str
a5 = True        # bool
a6 = (3,4)       # tuple
isinstance(a6,tuple)

import numpy as np
b1 = np.array(2.0)            # numpy.ndarray, [float],标量类型跟python同步
b2 = np.array([1,2])          # [np.int64]，数组元素类型跟np同步(int64,float64)
b3 = np.array([1.2,2.3])      # [np.float64]，数组元素类型跟np同步(int64,float64)
isinstance(b1, numpy.ndarray)
isinstance(b1.item(), float)
isinstance(b2[0], np.int64)
isinstance(b3[0], np.float64)


'''------------------------------------------------------------------------
Q: 有一个包含N个不同类型元素的序列(list/dic/str)，如何把它分解为N个单独的变量？
这个叫做解包，解包有2种，一种自动解包，一种手动解包(*/**)
'''

# 用变量自动解包
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

# 用*/**进行解包
lst = [1,2,3]
[*lst]                  # 解包后变成散的数，需要在list包起来分配内存
lst2 = [10,20,*lst,30]  # 解包后变成散的数，需要在list包起来分配内存

dct = dict(a=1,b=2)
{**dct}           # 解包后变成散的数，需要在dict包起来分配内存
dict(**dct)      # 解包后变成散的数，需要在dict包起来分配内存

def aa(m,n,l,a=1,b=2):
    print(m,n,l)
    print(a,b)
aa(*lst, **dct)  # 解包后提供给函数作为参数


'''------------------------------------------------------------------------
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


'''------------------------------------------------------------------------
Q: 有一组数据，如何获得最大的或最小的N个元素？
'''
num = [1,8,2,23,7,-4,18,23,42,37,2]
max_3 = sorted(num)[:3]   # 通过先排序再切片的方式获得最大/最小的N个元素
# 这种写法非常有意思，可以方便获得从小到大排序和从大到小排序
min_to_max = num.sorted()[::]
max_to_min = num.sorted()[::-1]

import numpy as np
min_to_max_index = np.argsort(num)[::]
max_to_min_index = np.argsort(num)[::-1]

'''------------------------------------------------------------------------
Q: 想创建一个字典，并且希望字典内保持输入时的顺序，而不是默认的key字母顺序？
'''
from collections import OrderedDict
d = OrderedDict()   # 通过OrderedDict类创建一个带顺序的字典，按添加顺序保存
d['foo'] = 2
d['bar'] = 1
d['spam'] = 3
for key in d:
    print(key, d[key])


'''------------------------------------------------------------------------
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


'''------------------------------------------------------------------------
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


'''------------------------------------------------------------------------
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


'''------------------------------------------------------------------------
Q: 有一组序列，我知道怎么排序，但不知道怎么筛选出特定要求的元素？
- 列表推导式做筛选
- 字典推导式做筛选
'''
import numpy as np

# 元素判断式筛选 (只能在nparray使用，在list中用列表推导式和字典推导式代替该方案进行筛选)
from numpy import random
random.seed(11)
arr = random.randint(10, size=(5,5,5))
arr[...,0]               # 切片0层
arr[...,0]>5             # 0层元素判断，返回同样size的bool变量
arr[...,0][arr[...,0]>5] # 通过判断筛选出具体值，作为list返回

random.seed(22)
arr2 = random.randint(10, size=(2,3))
arr2[0,:]
arr2[0,:]>4
out1 = arr2[0,:][arr2[0,:]>4]

out2 = arr2[0,:][[1,0,0]]               # 这种写法是提取法
out3 = arr2[0,:][[True, False, False]]  # 这种写法是筛选法，此时True/False与1/0不等效

flag = [n>1 for n in range(3)]  
flag
arr = np.array([10,2,333])
arr[flag]                               # 筛选法

# 列表推导式筛选
mylist = [1,4,-5,10,-7,0,3,-1]
newlist = [n for n in mylist if n > 0]     # 用列表推导式做筛选-格式1(先循环后判断)
newlist2 = [n if n>0 else 0 for n in mylist] # 用列表推导式筛选-格式2(先判断后循环)
# 列表推导式 [n for xx if xxx] 先for循环，再条件判断
# 列表推导式 [n if xx for xxx] 先条件判断，再进行for循环
# 原则上是相同的，但细微差别。。。。

index_gtz = np.where(np.array(mylist)>=0)  # 用numpy的where()函数可以返回index
index_ltz = np.where(np.array(mylist)<0)

# 字典推导式
d1 = dict(a=1,b=2,c=4,d=9,e=3)
d1-2 = {key:value for key,value in d1.items() if value > 4}



'''------------------------------------------------------------------------
Q: 更广阔的空间理解列表推导式 + 元素条件判断句？
1. 列表推导式核心是：一组数字能够转换为(0,1,2,3,4,5...)的变体，就能用列表推导式写出
2. 实际上有列表推导式/字典推导式
3. 元素条件判断可以加在任何元素后边（）
'''
# 列表推导式
lst = [n**2 for n in range(5)]

# 字典推导式
d1 = dict(a=1,b=3,c=5,d=7)
{key:value**2 for key, value in d1.items() if value > 4}

# 对单个元素条件判断
from numpy import random
flip = True if random.uniform(0,1) < 0.5 else False



'''------------------------------------------------------------------------
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



'''------------------------------------------------------------------------
Q: 为什么有时候复制一个变量会影响另一个变量，有时候又不会影响？
'''
# 这是一个重要知识点：在算法编写过程中，已经踩坑2次，每次耗费我2天都找不到问题根源
''' 0. 复制就是复制元素，而不是容器，关键看元素是可变对象还是非可变对象。
    1. python中一类叫不可变对象，包括数字/字符串/元祖，他们的复制只能创建新内存，所以是不会影响
    2. python中另一类叫可变对象，包括数组/字典，他们常规复制都是浅复制，不同变量名指向同一内存地址
    3. 相应的在函数形参上，不可变对象形参是传值，函数体内改变该对象，不会影响函数体外的原始变量；
       而可变对象形参是传址，即传递的是指针，在函数体内改变该对象，会影响函数体外的原始变量。
    4. 对于b=a[:], b=list(a), b=a*1, b=copy.copy(a)四种方式复制列表，如果列表没有嵌套，这几种方式相当与取到值，所以算深拷贝，
       结果都可以得到一个新的列表
       
       最可靠的深复制方式有3种：
    1. 采用额外的copy库，带有命令copy.deepcopy()
        import copy; copy.deepcopy(lst); 这是python中深复制，无论对象是否有嵌套
    2. 直接取值到不可变对象(数字，字符串，元组)，这样只能深复制。
       比如对于numpy，由于保存的都是非嵌套的数据，所以采用data.copy()函数都是深复制。
       numpy中可以通过data.flags查看OWNDATA选项，如果是False则是指针，如果是True则是自己有内存
'''

old = [4,1,3,['age',10]]
new_1 = old
new_1.append(100)
print(old, new_1)   # 用名称的形式拷贝，只是指针拷贝，指向同一内存，共同变化

old = [4,1,3,['age',10]]
new_2 = old[:]      # 用old[:]的形式是浅拷贝，对外层的所有对象都可以拷贝了
id(old),id(new_2)        # 两个主变量id不同，说明两个变量内存地址不同，
id(old[3]), id(new_2[3]) # 但两个包含的子数据list的id相同，说明是浅拷贝


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

# 再看numpy中: 由于numpy只能处理数值，也就是所有元素都是不可变对象，也就是深度复制
old = np.array([[1,2,3],[4,5,6]])
new = old.copy()     # 复制数字，所以是深拷贝
new[0,0]=100
print(old, new)


'''------------------------------------------------------------------------
Q: 如何生成正态分布的随机数据？
类似函数比较多，写法不统一，比较乱，numpy这块有点反人类的感觉
第一类：从(0,1)分布写起，可扩展到其他分布
    > rand(m,n) 固定在[0,1)的均匀分布，也可单个数rand()
    > randn(m,n) 固定在[0,1)的正态分布,也可单个数randn()
    
第二类：从便捷范围写起，这类函数使用得似乎更方便！！！
    > randint(low, high) 整数范围取1个数，或size个数
    > uniform(low, high) 实数范围取1个书，或size个数

核心功能2： 随机打乱序列
random.permutation(lst)
核心功能3：从一组数随机选一个
random.choice(lst)
'''
import numpy as np
# -------------区分分布类型，按尺寸(m,n)：rand, randn-----------------------
# 均匀分布
np.random.rand(2,4)    # 一组实数，只能取默认range=[0,1)
np.random.rand()       # 单个实数,[0,1)
# 正态分布
np.random.randn(2,4)   # 一组实数，范围[0,1)，正态分布
np.random.randn()      # 单个实数，范围[0,1)，均匀分布
r2 = 5*np.random.randn(2,4)+10  # 生成2x4的正态分布N(10,5)

# -------------区分数据类型，按范围(low,high)：randint, uniform-----------------------           
# 生成均匀分布数组（整数），[low,high)左闭右开
np.random.randint(0,5,size=(2,4))  # 生成[0,5]之间整数的均匀分布，random_integers()已被废弃
np.random.randint(0,5) # 一个整数(均匀分布)，不写size就是一个        
np.random.randint(2)  # 0,1之间取(0可省略不写，2是开口，所以是0,1两个数)             

# 生成均匀分布实数
np.random.uniform(0,10,size=(2,4))
np.random.uniform(0,10)


# seed只能保证单次有效,设置了seed就代表系统会指定一个随机算法的开始值，所以得到的第一个随机数相同。
np.random.seed(11)  # 指定seed(11),则每次都从25开始，每次往下的随机数都是固定顺序的
for i in range(10):
    print(np.random.randint(0,100))

for j in range(10):
    np.random.seed(11) # 指定seed(11),则每次都从25开始，且这个起始值不变
    print(np.random.randint(0,100))

# 序列随机打乱用permutation
# (random.shuffle的功能一样，但shuffle是implace操作且没有返回值，所以用得少）
np.random.permutation([1,2,3])  # 输入一个list，输出一个打乱的list
np.random.permutation(5)   # 可简写为5代表0开始的5个数，相当于range(5)
np.random.permutation(np.array([[1,2,3],[4,5,6]])) # 如果是多维矩阵，却只能随机打乱行，对元素和列无影响

# 从指定list中随机选一个数
lst = [1,10,100,1000]
np.random.choice(lst)

'''------------------------------------------------------------------------
Q: 如何把只含一个元素的array或者list转化为标量的float/int？
'''
# 先用type()函数查看数据类型，到底是矢量nunpy/list，还是标量float/int
type(x)  


'''------------------------------------------------------------------------
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



'''------------------------------------------------------------------------
Q. 如何理解和使用高阶函数map, reduce？
- 两者功能上有类似地方，都是2个input， 一个fn,一个list
map(映射)用于把fn map给每个list元素计算输出一个list，
reduce(浓缩)用于把fn map给每个list元素，但每一轮都是基于上一轮输出和本轮输入组合计算，最后输出一个标量
参考：https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/00141861202544241651579c69d4399a9aa135afef28c44000
'''
input = [1,2,3,4,5]
def fn(x):
    return x**2
out = list(map(fn, input))  # map返回的是一个迭代器

from functools import reduce
input = [1,2,3,4,5]
def fn(x,y):
    return x+y
out = reduce(fn, input)   # reduce返回一个标量

'''------------------------------------------------------------------------
Q. 如何理解和使用高阶函数partial？
在mmdetection的anchor generator中有使用这个partial函数。
'''
def addnums(x1,x2,x3):
    return(x1+2*x2+3*x3)
    
from functools import partial
newfunc = partial(addnums, 100)  # 新函数定义，相当与把地一个kwards定义下来
                                 # 原则是先定义的是kwards，

out = newfunc(2,3)   # 默认是固定了第一个参数


def test(a,b,c,d):
    print(a,b,c,d)
from functools import partial
newtest = partial(test, b=10, d=20)  # 通过关键字参数可以预先固定中间的参数
newtest(1,c=2)   # 调用的时候由于b,d已被定义成关键字参数，中间的c也只能基于关键字参数调用


'''------------------------------------------------------------------------
Q4. 如何进行堆叠和展平？
核心概念：
    1. concatenate, stack只能输入array，不能输入list，而torch.cat可以输入list/array
    2. concatenate通常用于处理二维数据，且size尺寸可以不同，虽然能够处理1维但只能一维扁平堆叠
    3. stack通常用于处理一维，且在一维数据下也能axis=0/1两个方向，且自动输出升为2维。虽然能够处理2维但要求输入size相同。
    4. 一般来说：一维数据用stack处理(胜在两个axis都可操作)，二维以上用concatenate(胜在不同size也能堆叠)
       一般来说：普通数据用stack/concatenate，tensor数据也一样用stack/cat
------------------------------------------------------------
'''
import numpy as np
# 展平
a = np.array([[2,1,4,5],[4,2,7,1]])
a.ravel()     # 在原数据上直接展平
a.flatten()   # 展平，但不影响原数据
# concatenate堆叠
b0=np.array([[1,2],[3,4]])
b1=np.array([[5,6],[7,8]])
b3=np.concatenate((b0,b1),axis=0)  # 最核心常用的堆叠命令(axis可以控制堆叠方向)

d0 = np.array([1,2,3])
d1 = np.array([4,5,6])
d3 = np.concatenate((d0,d1),axis=-1)


c0=np.array([[1,2],[3,4]])
c1=np.array([[5,6]])
c2=np.array([[5,6],[7,8]])
c3=np.concatenate((c0,c1),axis=0)  # 堆叠的数据可以是不同size的，但要保证空间上摆放的对应维度相同
                                   # 或者要理解成拼接维度不同，但其他维度都要相同：axis=1方向维度不同，但axis=0/2/..n维度要相同
c4=np.concatenate((c0,c1),axis=1)  # 报错！ 因为(2,2)与(1,2)如果在axis=1拼接，axis=0的维度需要相同。这种低维的肉眼也能看出来axis=1方向摆放不成
c5=np.concatenate((c0,c2),axis=1)  # ok
# stack堆叠
# stack在处理一维堆叠的直观效果很好，类似在两个轴上分别叠加，一维直观效果比concatenate好（concatenate不能处理一维axis=1的情况）
# 但stack本质上是先升维再堆叠：
a0 = np.array([1,2,3])
a1 = np.array([4,5,6])

a2 = np.stack((a0,a1),axis=0)   # stack()可以用于一维数据的多轴操作，且自动完成升维
a3 = np.stack((a0,a1),axis=1)  #stack多轴都可操作，自动升维

b0 = np.array([[1,2],[3,4]])  
b1 = np.array([[5,6],[7,8]])

# stack处理多维
b2 = np.stack((b0,b1),axis=0)  # 原来(2,2)升维到(1,2,2), 两个(1,2,2)的axis=0轴堆叠成(2,2,2)
b3 = np.stack((b0,b1),axis=1)  # 原来(2,2)升维到(2,1,2)，两个(2,1,2)的axis=1轴堆叠成(2,2,2)
b4 = np.stack((b0,b1),axis=2)  # 原来(2,2)升维到(2,2,1)，两个(2,2,1)的axis=2轴堆叠成(2,2,2)


d0 = np.array([[1,2,3],[4,5,6]])
d1 = np.array([[1,2,3],[4,5,6]])
d2 = np.array([[1,2,3],[4,5,6]])
d4 = np.stack((d0,d1,d2), axis=0)  # (2,3)升维到(1,2,3),再在axis=0方向concatenate成(3,2,3)
d5 = np.stack((d0,d1,d2), axis=1)  # (2,3)升维到(2,1,3),再在axis=1方向concatenate成(2,3,3)
d6 = np.stack((d0,d1,d2), axis=2)  # (2,3)升维到(2,3，1),再在axis=2方向concatenate成(2,3,3)

d7 = np.concatenate((d0,d1,d2), axis=0)  # 不升维，直观在行方向上组合
d8 = np.concatenate((d0,d1,d2), axis=1)  # 不升维，直观在列方向上组合


# 进一步看一下应用最广的一维的情况总结：
a1 = [1,2,3,4,5]
a2=[6,7,8,9,10]

a3 = np.stack((a1,a2), axis=0)      # stack直观：行变换方向堆叠
a4 = np.stack((a1,a2), axis=1)      # stack直观：列变换方向堆叠

a5 = np.concatenate((a1,a2),axis=0)  # concatenate效果不直观，实际上是两行数据串接成一行
a6 = np.concatenate((a1,a2),axis=1)  # concatenate报错

# 二维情况正相反：concatenate直观，stack虽能做但效果不直观
# 所以一维用stack,二维用concatenate



'''------------------------------------------------------------------------
Q4. 如何进行数据增加维度的操作？
'''
# 利用+号重载运算符的叠加功能，先对维度进行组合
[2,3] + [4,]   # [2,3,4]
(4,3) + (2,)   # (4,3,2)
(4,3) + (2)    # 报错：单维度的数据最好加一个逗号
#-----对array：用numpy的函数--------
