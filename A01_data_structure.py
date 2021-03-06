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


"""-----------------------------------------------------------------------
Q. 如何进行简洁的numpy数组创建
注意：只有如下2种情况参数是用tuple或者list传进去
1. np.ones((m,n)),np.zeros((m,n))这一类：是把size放在tuple传进去
    注意：np.random.rand()这类的size是直接传，而不是tuple，并且在pytorch中也把ones/zeros这层括号去掉了。
    说明numpy的这部分函数有点反直觉反人性
2. np.stack([a,b]), np.concatenate([a,b])这一类：是把多个array放在一个list传进去，所以是用[arr1,arr2..]
"""
import numpy as np
# 要加括号的 
a = np.array([[1,2],[3,4]])
np.zeros((2,3))
np.ones((2,3))
np.empty((2,3))

np.zeros_like(a)
np.ones_like(a)
np.empty_like(a)

# 不要加括号的：其他大部分都不用加了，包括random和pytorch里边所有的
np.random.rand(2,3)
np.random.randn(2,3)

# range的加强版：np.arange直接生成array
np.arange(0,10,2)   # (0,10)之间，间隔2
np.linspace(0,10,7) # (0,10)之间，取7个数



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

# %%
'''Q: 如何理解常规切片的写法？
    - 核心两个冒号的写法：lst[a:b:c]的意思为a代表起点，b代表终点，c代表间隔
    - 简写一个冒号的写法：lst[a:b]
    - 扩展出无数变形：lst[::]代表复制, lst[::-1]代表逆序, lst[:3]/lst[3:]代表从序号3分成2部分，lst[:-1]代表到最后一个数，
    - 注意：切片任一行或任一列，出来的都是一维(n,)，而不再是二维的行或者列.
    
从0开始并且不考虑最后一个元素的方式有如下优点：
1. lst[:3]就可以直接读出包含了3个元素，这跟range(3),np.arange(3)有相同功效，这个可以叫做去尾效应
2. lst[:3]和lst[3:]正好把list分成两部分，非常直观
3. lst[start:stop]可以通过起点和终点直接算出包含元素个数n=stop-start
'''
lst = [3,4,2,6,1,6,4]
print(lst[:3])   # 写成冒号3，正好代表有3个元素，非常自然
print(lst[3:]) # 写成3到最后，正好代表下一个元素到最后，从而把3当成分界点
# 把3当成分界点，冒号3代表包含3个元素，3冒号代表3个元素之后（参考流畅的python）

# 核心要记住a[a:b:c]中a代表起点，b代表终点，c代表间隔
# 以下是几个变种
a = np.array([0,1,2,3,4,5,6,7,8])
a[1:8:2]  # 表示位置号[1,3,5,7]
a[:8:2]   # 表示位置号[0,2,4,6]
a[1:8]    # 表示位置号[1,2,3,4,5,6,7]
a[1:-1:2] # 表示位置号[1,3,5,7]
a[::2]    # 表示位置号[0,2,4,6,8]

b = np.arange(20).reshape(4,5)
a[:, 1::4]  


# %%
'''Q. 更高级的切片和索引（冒号切片，数组索引）？
重点总结：
对于a = np.array([[1,2,3],[1,2,3],[1,2,3]])
    写成列表形式，代表的是抽取，如果行也是抽取，列也是抽取，则取两者交汇的点的值
            比如 a[[0,1,2], [0,1,2]]得到的是[1,2,3]
    写成冒号形式，代表的是筛选，如果行也是筛选，列也是筛选，则取两者共有的部分

注意：
    - 冒号的写法是切片，得到的是一片，比如一列或者一行
    - 列表/数组的写法是索引，得到的是某些点，所以行索引+列索引就是某些索引交叉点。
      但混合了行索引+冒号列则相当于切片了，比如a[[1,2,1],:]，比如a[[1,2],1:3]

a[m,:] 代表提取第m行
a[lst, :] 除了代表提取第m，n行，还需要按照m,n的顺序，相当于调整行顺序
a[arr, :] 也可以用1维array
a[arr1, arr2] 可同时筛选提取某几行，某几列，但要保证arr1/arr2为一维的，且长度相同
'''
a = np.arange(20).reshape(4,5)

# 用一个行索引列表来提取指定的行
a[(1,2,0),:]
a[[1,2,0],:]
a[0,:]        # 提取第0行
a[(1,1,0),:]  # 连续提取第1行2次和第0行
a[[0,2],:] = 1 # 提取其中2行
a[:, [2,3]] = 2 # 提取其中2列

# 用一个行索引数组(一维的)等效于行索引列表，用来提取指定某些行
a[np.array([1,3]), :]

# 同时提取行和列: 此时需要两种提取对应位数相同，也就是一个行对应一个列。
a[[1,3], [2,4]]  # 提取的是a[1,2]和a[3,4]
a[[2,2,0,3], range(4)]

# 用索引列表+标量索引
a[[1,2,3], 4]   # 提取的是第4列上的行索引号

# 组合使用：列表索引+切片写法，此时是先通过索引提取1,2,3行，然后切片0,1,2列，得到的是3行3列
# 这里需要区分a[[1,2,3],[0,1,2]]这种二维索引的方式，得到的是3个元素
a[[1,2,3], 0:3]     # 得到3行3列
a[[1,2,3], [0,1,2]] # 得到3个元素

a[[1,1,0], :3]

# %%
'''Q: list里边嵌套list是算法里边常见的一种数据结构，如果对这类list进行切片？
由于没有二维切片可以用，所以要制造出二维索引。
(重要技巧)用一个for循环，配合列表推导式，来得到切片。注意for循环不要取取元素，而是用元素位置号作为索引这样更灵活。

'''
lst = [[1,2,3],[4,5,6],[7,8,9]]
for i in range(len(lst[0])):
    featList = [sample[i] for sample in lst]  # 循环是取得list列的标准写法




# %%
'''Q: 创建dict字典最重要的2种方法?
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

# 增加
d1 = dict(a=1,b=2,c=3,d=4)
d1['e'] = 5  # 增加key/value
d1['b'] = 9  # 修改value

d1.update(h=5)  # 增加key/value，相对来说dict(a=1), d.update(a=1)这两个命令都是dict常用，因为比较书写简单。

d1.get('a')
d1['f']        # 第1种获得dict键值方法：如果键不存在就会报错，导致程序终结，不是特别好。
d1.get('f')    # 第2种获得dict键值方法：用get不会报错，没有就返回None。可用于判断是否None，不必终止程序

d1.setdefault('a', 2)  # 第3种获得dict键值方法：如果键存在就提取，如果键不存在就创建并设置为指定值
d1.setdefault('h', 2)

# 删除
d1.pop('a')

# %%
"""常用的判断一个变量是否存在的方法? locals() + get()
1. 用locals()可以返回一个所有变量汇总的字典，然后就可以用get()函数搜索字典里边的变量，这样即使不存在也不会报错被迫中断程序。
2. 
"""
if not locals().get(xyz, None):
    print('xyz exist, you can use it.')


# %%
'''Q.怎么使用OrderedDict和defaultDict?
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
    >包括np.float64, np.int32, np.float32...
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
这个叫做解包，解包的本质是分解成独立的item了，所以解包后的元素不占内存，也就不能直接使用，只有如下2种情况可用：
    >用在func中做参数，
    >解包后在外边加容器包起来，比如加list,tuple,[],()...
解包有2种方式，一种自动解包，一种手动解包(*/**)：
0. 自动解包：用对应位置，比如a,b,c = [1,2,3] 或比如 a, *b = [1,2,3]
1. 用'*'可以为所有迭代对象解包
2. 用'**'可以给字典解包
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

# 用*给所有迭代对象解包
lst = [1,2,3]
[*lst]                  # 解包后变成散的数，需要在list包起来分配内存
lst2 = [10,20,*lst,30]  # 解包后变成散的数，需要在list包起来分配内存


# 用**给字典解包
dct = dict(a=1,b=2)
{**dct}           # 解包后变成散的数，需要在dict包起来分配内存
dict(**dct)      # 解包后变成散的数，需要在dict包起来分配内存

def aa(m,n,l,a=1,b=2):
    print(m,n,l)
    print(a,b)
aa(*lst, **dct)  # 解包后提供给函数作为参数


# %%
"""如何合并两个字典？
合并两个list很简单，lst1 + lst2
但合并两个字典之前没有特别好的方法，现在有了，如下。
"""
# 最快的方法
d1 = dict(a=1, b=2, c=3)
d2 = dict(a=2, d=4, e=5)
d3 = {**d1, **d2}        # 注意，如果两个字典中有相同的键，写在前面的字典的键会被后边的字典的键覆盖掉。


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
min_3 = sorted(num)[:3]   # 通过先排序再切片的方式获得最大/最小的N个元素
# 这种写法非常有意思，可以方便获得从小到大排序和从大到小排序
min_to_max = sorted(num)[::]
max_to_min = sorted(num)[::-1]

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




# %%
"""Q. 区分逻辑运算符
    1. 如果是逻辑变量的操作，and/or跟&,|是一样的：比如(3 > 1) & (3 < 1)
    2. 如果是对单个数值的操作，&,|是表示
    3. 在python的原生数据类型中，只能针对单个逻辑变量或者单个数值进行 & |
       或者采用专门的set数据类型，进行相应的& |
       但注意两个list无法进行元素级别的& |
    
    3. numpy的优势
"""

# %%
"""如何获得一组数据的不重复数据集？
1. 认识set数据格式：一种专门的用于集合运算的数据格式，默认就是不重复元素。
    创建set: set(), 或者{2,1,3,4}, 类似于c++中数组的创建方式。 
    添加元素：a.update()  这个类似与dict的方法dct.update(c=4)
    集合运算：能够非常方便进行相与相或等操作，但只能用set格式进行操作
        - a - b 
        - a | b 或
        - a & b 与
        - a ^ b 异或

2. 获取不重复数据方式：
    - 转为set，这种方法稍有麻烦，因为set()之后返回的不是array/list之类的数据结构，而是set()类型，需要再转回list/array
    - 用np.unique(), 这种方法很常用，非常方便
"""
a = {1,2,3,4,5}
b = {3,4,5,6,7}
a - b    # a中去除b
a | b    # 或： 最大范围
a & b    # 与： 同时包含的元素
a ^ b    # 异或：

c = set('hello yoo')  # 会类似对list的操作一样，把字符串分拆成一个个字符
c.update('world')    # 把新元素加进原来的set中，并且是分拆成字符后加入

d = set(('hello', 'world'))  # 如果是添加多个字符串作为元素个体，则需要用内括号扩起来添加。
d.add('nice')       # 此时添加一个字符串只能用add，不能用update, 因为update会分拆字符串

# 方式1：用专门的集合数据格式set，专门解决集合元算：并集，差集，交集，因为set()默认就是不重复
lst = [2,1,3,4,2,4,1,4,2,4,5,2,3,1]
out = list(set(lst))

# 方式2: 用numpy的unique函数
aa = np.array([2,3,4,1,2,1,4,3,2,3,4,2])
np.unique(aa)


'''------------------------------------------------------------------------
Q: 有一组列表，希望找到序列中出现次数的元素，甚至每个元素的出现次数，怎么做？
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
Q: 有一组字典，希望对某个键进行排序，用sorted（）怎么定义？
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
1. 如果是numpy直接用元素判断式筛选arr[arr>i], 无需写循环
   或者用np.where筛选index
2. 如果是原生的python数据结构list.dict，则需要利用列表推导式/字典推导式（循环判断筛选）
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



# %%
"""Q. 筛选之后的元素替代？
用np.where最方便
"""
a = np.array([3,1,3,0,1,4,5])
idx = np.where(a==1)             # np.where()的第一种用法：筛选index   

b = np.where(a==1, 100,-100)     # np.where()的第二种用法：筛选之后的直接替代。这个方法非常好用



# %%
"""Q. 如何对数组做padding?
numpy提供了一个现成函数np.pad(arr, value, mode)
- 其中value是对每个轴的填充值，比如((1,1),(1,1))表示在第0轴前面填充
"""
# 对普通二维数组的边缘填充
a = np.ones((3,3))
np.pad(a, ((1,1),(1,1)), "constant")  # 表示在0轴前后各1行0,在1轴前后各一列0 
np.pad(a, ((2,1),(1,2)), "constant")  # 表示在0轴(行)前面填充2行，后边填充1行，在1轴(列)前面填充1列后边填充2列。
np.pad(a, ((2,1),(1,2)), "constant", constant_values=(10,20))  # 指定前面行和列填充10， 后边行和列填充20

# 对高维数组的边缘填充
b = np.ones((2,4,3,3))
np.pad(b, ((0,0), (0,0), (1,1), (1,1)), "constant")  # 表示对0轴1轴不操作，对2轴3轴前后都加一行或列的0


# %%
'''Q: 相比于list, 似乎字典用得比较少，而且似乎[('eason',6),('jack',4)]这种tuple结构可以替代字典。
但什么时候用字典方便？字典的方便特性怎么用？
1. 在带有字符/数字混合的情况下，用字典更好，因为字典天然对字符可以筛选：d1['str']。
   而用[tuple]结构，对筛选字符就得上for循环，不是太方便
2. 字典的基本功能很完善，不用担心使用不方便：
    > 创建
    > 切片
    > 排序：list所有方法都可用在d1.values()上
    > 筛选：天然可以筛选字符串，还可以用字典推导式强大支持
'''
# 字典的创建：用dict已经很方便了
d1 = dict(a=1,b=9,c=-2,d=5)

# 字典的切片提取: 需要基于key
d1['a']

# 字典的排序: 
sorted(d1.values())  # 单独排序values
sorted(d1.keys())    # 单独排序keys
sorted(zip(d1.values(), d1.keys()))  # 带着key，排序value

# 字典的筛选：字典推导式
d1['a']    # 字典的天然筛选属性就是key
{key:value for key, value in d1.items() if value ==-2}  # 推导式永远是python最强筛选利器：普适于dict/tuple/dict
{key:value for key, value in d1.items() if value >0}


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
几个专门针对set的函数：union()表示并集, ^表示差集， 两者都只能用a.union(b), a^b这种写法，而不是函数的写法。  

注意：为什么只有并集和差集的运算符，而没有交集运算符？因为有了并集和差集，可以直接求交集，但如果有了并集和交集，
为了另一部分面积还是需要求差集，所以为了简洁，只实现了并集和差集，交集操作就用(并集与差集)差集就可以算出来了。

1. 并集(或)：两个数列合并不重复的数据a.union(b)
2. 差集(异或)：两个序列不相同的数据，用a^b
3. 交集： 并集与差集的差集

'''
a = [1,2,3,4]
b = [1,4,5]

ret_list = [item for item in a if item not in b]  # a对b的差集
ret_list = [item for item in a if item in b]  # 交集

# union()
set(a).union(set(b))  # union只能针对set来使用
set(a)^set(b)


# 高级写法求差集，并集，交集
ret_list = list(set(a_list)^set(b_list)) # 差集: 用 ^ 
ret_list = list(set(a_list).union(set(b_list))) # 并集： 用a.union(b)
ret_list = list((set(a_list).union(set(b_list)))^(set(a_list)^set(b_list))) # 交集



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
本质只要记住4个函数：random.randn(2,4) 标准正态分布N(0,1)值 -> 可生成各种取值范围的正态分布
                    random.rand(2,4) 均匀分布(0,1)值 -> 可生成各种取值范围的均匀分布
                    random.randint(0,10,size=(2,4)), 整数值  -> 可生成各种类型整数
                    random.permutation(lst), 随机乱序 -> 可替代random.shuffle, random.choice

核心功能1：随机生成
类似函数比较多，写法不统一，比较乱，numpy这块有点反人类的感觉, 但最终总结成好理解的2类：
一类按分布的rand/randn, 一类按取值范围的randint/uniform
或者再进一步总结成3个
    np.random.randn(2,4): 标准正态分布(均值0方差1)
    np.random.rand(2,4): 均匀分布的实数(取值0到1之间)
    np.random.randint(low,high, size=(2,4)): 均匀分布的整数
(而np.random.uniform(1,10, size=(2,4))是可以用10*np.random.rand(2,4)替代掉的，所以在pytorch中取消了这个函数的对应函数) 

第一类：从分布率写起，可扩展到其他分布  (对应pytorch有torch.rand()/torch.randn())
    > rand(m,n) 均匀分布u(0,1)，也可单个数rand()
        通过变换可以得到(-1,1)或任何其他区间的均匀分布，比如rand(2,4)*2-1就是(-1,1)的均匀分布
    > randn(m,n) 标准正态分布N(0,1),注意不是取值0-1而是均值和方差是0/1，也可单个数randn()
        通过变换可以得到其他区间的正态分布
    
第二类：从取值范围写起  (对应pytorch只有torch.randint())
    > randint(low, high) 整数范围取1个数，或size个数
    > uniform(low, high, size=(m,n)) 实数范围取1个数，或size个数，默认low=0,high=1

核心功能2： 随机打乱序列  (对应pytorch只实现了一个torch.randperm()对从0开始的数组重排列)
random.permutation(lst), 如果是放入array，则每行看成一个元素，也作为list按行随机乱序
random.shuffle(lst)    inplace操作
核心功能3：从一组数随机选n个  (对应pytorch没有)
random.choice(lst, size=3, replace=True), 可随机输出size个数，默认replace=True说明数字抽样会放回，也就是会有重复数字。可设置为不放回抽样replace=False

总结下来随机打乱顺序只需要1个函数:
    np.random.permutation()
(而random.shuffle()可以用permutation完全替代;
 而random.choice()也可以用permutation打乱index后提取任意个来替代： idx = np.random.permutation(range(len(data)))[0]
'''
import numpy as np
import matplotlib.pyplot as plt
# -------------区分分布类型，按尺寸(m,n)：rand, randn-----------------------
# 均匀分布
np.random.rand(2,4)    # 一组实数，只能取默认range=[0,1)
np.random.rand()       # 单个实数,[0,1)
# 正态分布
np.random.randn(2,4)   # 一组实数，范围[0,1)，正态分布
np.random.randn()      # 单个实数，范围[0,1)，均匀分布
r2 = 5*np.random.randn(2,4)+10  # 生成2x4的正态分布N(10,5)

# 检查分布类型
a = np.random.rand(50000)
b = np.random.randn(50000)
plt.subplot(121)
plt.hist(a, 20, range=(-3, 3), label='rand')  # 显示的均匀分布
plt.legend()

plt.subplot(122)
plt.hist(b, 20, range=(-3, 3), label='randn') # 显示的是标准正态分布
plt.legend()


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

t = [2,1,3,7,5]    # 虽然是np的函数，但支持list
np.random.shuffle(t) # inplace操作

# 从指定list中随机选一个数
lst = [1,10,100,1000,2,20,200,2000]
arr = np.array(lst)
np.random.choice(lst, size=2)   # 默认
np.random.choice(arr)



'''------------------------------------------------------------------------
Q: 如何把只含一个元素的array或者list转化为标量的float/int？
'''
# 先用type()函数查看数据类型，到底是矢量nunpy/list，还是标量float/int
type(x) 
# 然后切片取出
x[0] 

# %%
"""Q.如何判断空list/空dict/空元组？
1. 空list/空dict/空元组都相当与False，可以很方便的直接判断True or False
"""
aa = {}
if aa:
    print('aa is not empty.')
else:
    print('aa is empty')
    

# %%
'''Q: 如何判断None这种变量？
1. None/0/空列表[]/空元组()/空字典{}都可以等效于False
   所以可以直接用True/False的方式判定

2. 也可以把None/True/False看成对象， 用is/is not来判定
    这里注意，虽然可以用if a != None来判断，但最好还是用if a is not None来做，
    因为前一种方法在a为数组时直接报错，也就是说!=只能针对数值做判断，如果出现一个数组类型他就直接报错了。
    而后一种方法a无论是什么都不会报错，也都会完成判断。


'''
# 方式1, 当成布尔变量，直接判定，最为方便
bb = False
if bb:
    print('bb is True')
else:
    print('bb is False')

cc = None
if cc:
    print('cc is not None.')
else:
    print('cc is None')
    
# 方式2：当成对象，用is/is not判定，也ok
x = False
if x is False: 
    print('x is False')
if x: 
    print('x is True')

y = None
if y is None: 
    print('y is None')  # is None
if y is not None: 
    print('y is not None')


# %%
"""--------------------------------------------------------------------------
Q. 如何理解zip函数
1. zip()函数是把一个迭代对象转换成tuple输出
"""
a = [1,2,3]
b = [4,5,6]
z1 = zip(a)
z2 = zip(a,b)

next(zip(a))    # 输出的是一个zip元素，为单元素元组(1,)
next(zip(a,b))  # 输出的是一个zip元素，为多元素元组(1,4)
'''------------------------------------------------------------------------
Q. 如何理解和使用高阶函数map, reduce？
1.两者功能上有类似地方，都是2个input， 一个fn,一个list
    >map(fn, list)
    >map(映射)用于把fn map给每个list元素计算输出一个list
     注意：旧版python输出list，新版python输出迭代器，list()转换后就是list了, 或通过*解包map后也可以得到散的元素)
    >reduce(fn, list)
    >reduce(缩减)用于把fn map给每个list元素，但每一轮都是基于上一轮输出和本轮输入组合计算，最后输出一个标量
参考：https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/00141861202544241651579c69d4399a9aa135afef28c44000
'''
# map()的基础演示
input = [1,2,3,4,5]
def fn(x):
    return x**2
out = list(map(fn, input))  # map返回的是一个迭代器
out1 = [*map(fn,input)]
# map在multi_apply中的高级应用(参考mmdetection)
def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func # 用partial()函数先绑定kwargs
    map_results = map(pfunc, *args)                     # 对输入进行map()，返回一个map迭代对象，使用list可转化成对应的list
    return tuple(map(list, zip(*map_results)))          # 对输出map对象先解包，然后zip组合，然后zip转list，然后装入tuple


# reduce()的基础演示
from functools import reduce
input = [1,2,3,4,5]
def fn(x,y):
    return x+y
out = reduce(fn, input)   # reduce返回一个标量，相当于累加了input

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
    1. stack/concatenate的函数输入都是一个list形式[d1,d2,d3..]这非常方便：可以
       用list作为常规放置各种数组的容器，最后直接丢进stack/concatenate函数里边即可。
    2. concatenate通常用于处理二维数据，且size尺寸可以不同，虽然能够处理1维但只能一维扁平堆叠(axis=0)
       如果设置axis=1去堆叠一维数据则报错。所以一维的列方向堆叠要么先手动升维要么用stack自动升维堆叠
    3. stack: 升维堆叠。通常用于处理一维，且在一维数据下也能axis=0/1两个方向，且自动输出升为2维。虽然能够处理2维但要求输入size相同。
       axis=i就表示从左往右的i位置先增加一个新维度，然后在该维度叠加。
       比如针对一维数组(n,)的操作，如果axis=0, 相当于(1,n)进行堆叠得到(k,n)
       而如果axis=1，就等效于axis=-1, 则相当于(n,1)进行堆叠得到(n,k)
       从而实现了对一维数组的两个方向的随意堆叠，并且省去了做转置的操作。
       
       而stack对二维数组，一样也是先升维度，比如2个(2,3)的数组，在axis=0方向堆叠，得到的是(2,2,3)
       
    4. 一般来说：一维数据用stack处理(胜在两个axis都可操作)，二维以上用concatenate(胜在不同size也能堆叠)
       一般来说：普通数据用stack/concatenate，tensor数据也一样用stack/cat
       而实际情况却是：即使是高维堆叠，用stack也更多，因为高维堆叠都希望先升维度，比如2张img堆叠,就会升出一个叫batch的维度，用stack实现正合适
    
    5. pytorch的对应：
        np.stack() -> torch.stack()                                # 用在一维(升维堆叠)
        np.concatenate([a1,a2],axis=1) -> torch.cat([t1,t2],dim=1) # 用在二维(维度不变)
------------------------------------------------------------
'''
import numpy as np
# --------------展平--------------
a = np.array([[2,1,4,5],[4,2,7,1]])
a.ravel()     # 返回一个视图，跟原数据有链接，对视图修改会影响原始数据
a.flatten()   # 返回一个拷贝（不改变原数据），对拷贝修改也不会影响原始数据，相当于flatten返回的是脱敏数据

# concatenate堆叠: list/array都可，只不过一般用在二维的list/array上
a = [[1,2],[3,4]]
np.concatenate([a,a],1)
np.concatenate(a,0)
np.concatenate(a,1)  #

b0=np.array([[1,2],[3,4]])
b1=np.array([[5,6],[7,8]])
np.concatenate((b0,b1),axis=0)  # 最核心常用的堆叠命令(axis可以控制堆叠方向)
np.concatenate((b0,b1),axis=-1)

d0 = np.array([1,2,3])  # (3,)
d1 = np.array([4,5,6])  # (3,)
np.concatenate([d0,d1],axis=-1)  # -1表示


c0=np.array([[1,2],[3,4]])
c1=np.array([[5,6]])
c2=np.array([[5,6],[7,8]])
c3=np.concatenate((c0,c1),axis=0)  # 堆叠的数据可以是不同size的，但要保证空间上摆放的对应维度相同
                                   # 或者要理解成拼接维度不同，但其他维度都要相同：axis=1方向维度不同，但axis=0/2/..n维度要相同
c4=np.concatenate((c0,c1),axis=1)  # 报错！ 因为(2,2)与(1,2)如果在axis=1拼接，axis=0的维度需要相同。这种低维的肉眼也能看出来axis=1方向摆放不成
c5=np.concatenate((c0,c2),axis=1)  # ok
# stack堆叠
# stack在处理一维堆叠的直观效果很好，类似在两个轴上分别叠加，一维直观效果比concatenate好（concatenate不能处理一维axis=1的情况）
# 但stack本质上是先升维再堆叠： axis = i 就是先增加一个i维度，该维度=1, 
# 先改写数组为升维以后的数组，然后在该维度叠加
a0 = np.array([1,2,3])
a1 = np.array([4,5,6])

np.stack((a0,a1),axis=0)   # stack()可以用于一维数据的多轴操作，且自动完成升维: (3,) -> (1,3) ->堆叠变(2,3)
np.stack((a0,a1),axis=1)  # (3,)升维到(3,1)堆叠变(3,2)

# axis=-1的用法: 很适合理解一维的变化过程，一个(3,)的数组在最后追加一个维度为(3,1)然后n个堆叠为(3,n)，
# 非常自然的理解，非常顺滑的理解，得益于-1的认识以后，所有理解都通了
np.stack((a0,a1), -1)  # -1的含义是指定最后一个维度进行升维，从(3,)升维(3,1),然后在升维的维度进行堆叠

# stack处理多维
b0 = np.array([[1,2],[3,4]])  
b1 = np.array([[5,6],[7,8]])
np.stack((b0,b1),axis=0)  # 原来(2,2)升维到(1,2,2), 两个(1,2,2)的axis=0轴堆叠成(2,2,2)
np.stack((b0,b1),axis=1)  # 原来(2,2)升维到(2,1,2)，两个(2,1,2)的axis=1轴堆叠成(2,2,2)
np.stack((b0,b1),axis=2)  # 原来(2,2)升维到(2,2,1)，两个(2,2,1)的axis=2轴堆叠成(2,2,2)
np.stack((b0,b1),axis=-1)  # axis = -1跟axis=2效果一样

d0 = np.array([[1,2,3],[4,5,6]])
d1 = np.array([[1,2,3],[4,5,6]])
d2 = np.array([[1,2,3],[4,5,6]])
np.stack((d0,d1,d2), axis=0)  # (2,3)升维到(1,2,3),再在axis=0方向concatenate成(3,2,3)
np.stack((d0,d1,d2), axis=1)  # (2,3)升维到(2,1,3),再在axis=1方向concatenate成(2,3,3)
np.stack((d0,d1,d2), axis=2)  # (2,3)升维到(2,3,1),再在axis=2方向concatenate成(2,3,3)

np.concatenate((d0,d1,d2), axis=0)  # 不升维，直观在行方向上组合
np.concatenate((d0,d1,d2), axis=1)  # 不升维，直观在列方向上组合

# axis=-1的用法
np.stack((d0,d1), -1)  # axis=-1 就是指最后一个维度，这里原本2个维度，所以-1指下一个维度，
                       # 因此在一维stack中-1等效于axis=1, 二维stack中-1等效于axis=2
                       
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
Q. 如何进行数据的重复叠加？(堆叠是少量不同数组组合，重复叠加是大量相同数组组合)

1. 使用np.tile((m,n)) 同时在二维堆叠: 如果是一维数组横向复制不会升维度，如果是二维数组复制就会升维度。
2. 注意pytorch的对应有变化，变成repeat()了
    np.tile() -> t.repeat(m,n) # 双维度同时堆叠，所以基本上numpy/pytorch都用这行的这两个函数，比前一行的方便。

3. 注意区别np.repeat()这是用来对每个元素进行重复的函数，而不是堆叠的函数
   np.repeat(data, 3) #
'''
import numpy as np

# 用np.tile()实现同时在二维堆叠
c0=np.array([1,2,3,4,5])
np.tile(c0, (3,1))   # 用tile很方便实现行重复，列重复，且不会搞错，用repeat总会搞错
np.tile(c0, (3,))
np.tile(c0, (2,3))
np.tile(c0.reshape(-1,1), (2,3))

c1 = np.array([[1,2],[3,4]])
np.tile(c1, (3,2))


# 另一个类似函数np.repeat是元素重复： 注意repeat函数在处理(1,n)和(m,n)数组的重复时，行为非常特殊，需要仔细分析
a0 = np.array([1,2,3,4])
np.repeat(a0,3,axis=0)  # 相当于一维数组中每个元素重复4次
np.repeat(a0,3,axis=1)  # 报错：一维数组不能直接纵向元素复制

# 二维数组
a2 = np.array([[1,2,3,4]]) # 二维数组，第一维=1  (1,4)
np.repeat(a2,3, axis=0)    # 
np.repeat(a2,3, axis=1)

# 二维元素
b0 = np.array([[1,2],[3,4]])
np.repeat(b0,3,axis=0)  # 对于二维数组，repeat取出[]内的部分进行重复堆叠
np.repeat(b0,3,axis=1)


'''------------------------------------------------------------------------
Q. 常用最简单的几个打印命令？
两种打印输出的格式，其中格式1少写好多字符，还是用格式1方便
1. print的格式化输出方式1： 用 %格式 代表一个变量和格式，然后在字符串后边用%name写出变量名
    - 'a = %s'   代表字符串
    - 'a = %d'   代表整数
    - 'a = %.3f' 代表3位小数的浮点数
    - %a, %(a, b) 代表后道的变量

2. print的方式2： 用{:格式} 代表一个变量和格式，然后后边用.format(name)写出变量名
    - 'a = {:.3f}'.format(a)
'''
# 采用format的形式格式化输出： '{格式1}{格式2}'.format()
# {:s}为字符，{:f}为浮点数，{:.6f}为6位浮点数，{:d}为整数
from math import pi
s1 = 'pi = {:.6f}'.format(pi)  #写法{:.6f}代表了小数点后6位的float数据
print(s1)  
a = 5
b = 'eason'
s0 = '{:s} age is {:d}'.format(b,a)
print(s0)

# 采用%的形式格式化输出：  '%格式1%格式2'%(变量1,变量2)
# %d为整数，%f为浮点数(%.6f为6位浮点数)，%s为字符串
s2 = 'pi = %.50f'%pi  # 小数点后50位
print(s2)
a =5
b='eason'
s3 = '%s age is %d'%(b,a)
print(s3)

print('my friend %s can calculate %.6f!'%(b,pi))




# %%
"""Q. Numpy的排序筛选
本质的几句话：
numpy排序两种情况
    1. 直接排序元素，此时只能无关联性的各行或各列自己排序自己，用np.sort(arr, axis=1), 或者arr.sort()
    2. 先排序某列index, 再调整每行，用arr[np.argsort(arr[:,1])]， 这在一行一个样本的排序中及其有用。
    
numpy筛选两种情况：两种情况其实本质一样，都可以方便筛选元素或者按行，非常有用
    1. 先用不等式筛选出bool列表，然后再筛出任意行， 比如a[a>0]筛元素， a[a[:,1]>0]筛某些行
    2. 先用np.where(a)筛选出index列表，然后再筛出行，比如a[np.where(a[:,1]>0)]筛选某些行
    
具体来说------------------------------
numpy排序: 
    排序返回数值: np.sort(a)返回副本, a.sort(axis=1)在原数据上直接操作 (这个sort函数是numpy自己的，跟python的sort不一样)
    排序返回index: np.argsort(a)

numpy筛选：
    筛选返回bool list: a>0
    筛选返回value: a[a>0]
    筛选返回index: np.where(a>0)
numpy能够很好的处理筛选的问题，最常见的多条件筛选，最常用的方法是flag list，
通过不同的判断方式(f1>0, f2<-2之类)得到不同的flag分别代表不同条件结果，
然后f1 & f2 & f3..，即可得到最终的条件flag，最后data[flag]即可得到需要的数据。
非常方便。
"""
# list的排序和筛选
a1 = [2,1,4,3,6]
a2 = dict(a=1,b=3,c=5,d=7)
sorted(a1)      # list排序
d0 = [i for i in a1 if i >3]   # list筛选：列表式筛选
d1 = {key:value for key,value in a2.items() if value > 4} # 字典筛选： 字典式筛选

# array的排序
b1 = np.array([[2,7,0,5,8,4],
               [3,5,7,8,9,4],
               [0,8,1,4,2,2]])
np.sort(b1, axis=0)
np.sort(b1, axis=1)
b1.sort(axis=1)       # 

np.argsort(b1)                 # 排序 - 返回index

b1[np.argsort(b1[:,1])]   # 非常有用的排序方法：先得到某列index排序然后再排序每行样本


# array的筛选：可以返回3种类型的数据：bool/value/index，可以说非常灵活。
# 核心思想：筛选一般返回bool,因为bool很方便，跟切片可以结合a[bool]直接获得对应位置的数值
# 虽然index也有一样效果，但bool不需要记位置，显然更快捷，而inidex倒往往需要用函数才能获得
b1 = np.array([[5,3],
               [0,2],
               [4,6]])
b1[:,1] > 2                  # 直接判断筛选，返回bool list

# 以下两种方法类似，都很有用。方法一相对更方便。
b1[b1[:,0]>2]             # 先找到bool list，然后筛选每行
b1[np.where(b1[:,0]>2)]   # 先找到index list, 然后筛选每行

from numpy import random
a = random.randn(10)
a_bool = a > 0
b_bool = (a > 0) & (a < 1)

a[a > 0]
a[(a > 0) & (a < 1)]     # 把bool list往切片一丢，就能得到数值
b = np.where(a>0)
a[b]                     # 把index往切片一丢，也能得到数值，跟丢bool是一样的

"""但要注意，numpy的排序用np.sort会同时在每一个指定轴上排序，比如axis=0按列则每一列都要单独排序。
如果要想按照某一列的排列顺序而不去动其他列，比如一行一个样本的情况，则不能直接用np.sort。
"""
c = np.array([[2,5,1],
              [3,0,4],
              [0,6,2]])
    
np.sort(c, axis=0)        # 这种对元素的直接排序得到的是每一列元素都排序，不适合对于一行一个样本的这种样本排序
idx = np.argsort(c[:,0])  # 而这种argsort排序，却可以对任一列单独排序，且通过得到的index来按行调整顺序，非常适合对一行一个样本的排序
c[idx]  # 通过idx来对一行作为最小个体来排序，非常有用的技能。

# 多条件筛选实例
np.random.seed(6)
data = np.random.randint(-200,200,size=(4,10))

below_flags = np.zeros((data.shape[0],data.shape[1]))
below_flags = (data<10) & (data>=0)                            #产生条件标签1

above_flags = np.zeros((data.shape[0],data.shape[1]))
above_flags = (data>-10) & (data<=0)                           #产生条件标签2

all_flags = below_flags | above_flags                         #组合条件标签

data[all_flags]                                              # 通过标签显示数据


'''------------------------------------------------------------------------
Q. python中的位运算符
1. 位运算符： &(与), |(或), ^(异或), ~(取反), <<(左移动), >>(右移动)
2. 区别and/&: 其中&是按位与的操作，而and是逻辑运算符(and/or/)
'''




# %%
"""Q. 各种数据类型的相互转换
"""
import numpy as np
import torch
l1 = [1,2,3,4]

a1 = np.array(l1)   # list转array
l2 = a1.tolist()    # array转list



# %%
'''Q. Numpy的数学计算函数要么是采用math库对单个元素的计算，要么就是采用numpy()的计算函数
*注意：list不支持这些计算！！
1. 常规按位操作的计算函数：
2. 缩减(规约)操作的计算函数：
3. 相应的numpy数学计算函数：
'''
import numpy as np
a = np.array([1.2,-2.,3.5])
b = np.array([[1,0,2],[5,3,4]])

# numpy常规按元素操作 ------ pytorch中也完全照抄了这些函数和概念
np.exp(b)
np.log(b)    # 注意默认log是以np.e为底
np.log10(b)  # 除非注明np.log10()
np.log2(b)   # 除非注明np.log2()
np.abs(a)
np.add(b, 10)
b + 10
np.multiply(a, 10)   # pytorch是用mul
a * 10
np.ceil(a)
np.floor(a)
np.round(a)
np.power(a,2)    # 注意这个n次方的函数，跟python自带的次方函数不同(python自带的是pow(a,2),且只能针对单个数值计算)
np.sign(a)
np.sqrt(b)
np.sin(b)
np.maximum(b,b+3)    # 有一个按元素操作的maximum(), 不过用的少，大部分都是用缩减操作的max()
np.maximum(3, b)    # 按元素操作需要了解
np.minimum(-3,a)
a / a               # numpy把除号也重载为按元素操作了，非常统一，但注意list不支持按位操作
a * a               # numpy把乘号也重载为按元素操作了，非常统一，但注意list不支持按位操作


# numpy缩减(规约)操作 ------ pytorch也照抄了，只是把axis改成了dim
np.max(b, axis=1)
np.min(b, axis=1)
np.argmax(b, axis=1)
np.sum(b, axis=1)
np.cumsum(b, axis=1)

# numpy的比较函数

# 注意在pytorch中主要都是按位操作，所以torch.max(a,b)是按位操作
# 而np.maximum(a,b)才是按位操作，而np.max()是缩减操作。这在iou计算找lt/rb坐标时要注意，只能用按位操作


""" ----------------------------------------------------------------------
Q: Numpy中如何手动扩充维度？
1. 使用None
2. 使用np.newaxis
3. 使用reshape
"""
import numpy as np
# 一维扩充维度
a = np.array([1,2,3])    # 一维(3,)
a1 = a[:, None]          # 变(3,1)
a2 = a[:, np.newaxis]    # None等价于np.newaxis

a3 = a[None, :]          # 变(1,3)
a4 = a[np.newaxis, :]

# 多维扩充维度
b = np.array([[1,2,3],[4,5,6]])  # 二维(2,3)
b1 = b[:, None]                  # 变(2,1,3)
b2 = b[..., None]                # 变(2,3,1)

# reshape：其中-1参数代表无所谓多少
c = np.array([[1,2,3],[4,5,6]]) 
c.reshape(3,-1)   # 3行x列
c.reshape(-1)     # 单个-1参数代表无所谓，只要维度是(x,)也就是一行

d1 = np.array([1,2,3,4])  # (4,)
d2 = np.array([5,6,7,8])  # (4,)
np.stack([d1,d2],-1)      # 这里-1的含义是指定最后一个维度进行升维，所以是(4,1),然后在升维的维度进行堆叠


"""--------------------------------------------------------------------
Q: Numpy中自动扩充维度的广播机制是怎么样使用？
广播机制是指两个array进行运算时，通常按位操作，但如果维度不一致，就会自动进行维度扩充，这叫广播机制。
广播机制的应用场景：
    - 两组数据，比如(m,4)与(4,)，一组数据的每个样本(即每行)需要与另一组进行运算，比如相减，则需要对(4,)这个数据进行复制扩充，所以需要广播。
      这种情况符合自动广播的机制，所以无需手动广播。
    - 两组数据，比如(m,4)与(n,4)需要两两交叉相加运算(类似平移)，此时相当于(m,4)要跟每一个(1,4)做相加，最后得到n个(m,4)，也就是(n,m,4), 需要先对n的数据进行广播。
      这种情况不符合自动广播的机制，所以需要手动广播：(m,4)+(n,1,4)->(n,m,4), 手动广播的动作就是手动插入一个坐标轴，等效于提取n个样本中的一个，借用他进行自动广播。

广播机制的原则：
    >从后往前，进行维度比较
    >如果满足两个条件之一，就属于维度兼容，可以运算：(总结就是0变1,1变n)
     1.两个维度相等(显然可以运算)，2.其中一个维度=1(广播/自动维度扩充)
     也就是说维度为1的这个array，该维度的向量会自动复制到跟另一array的维度相等，从而可以相互运算。
     但要注意复制该维度从1到n相当于复制了该维度右边所有维度的数据(参看下面高维广播的实例)
    >其实应该还有一个条件，其中一个维度=0，这也是可以兼容的，此时0的维度也要复制扩充到另一维度维度
     或者可以把没有写的这个维度当成1，比如(5,)这就是(1,5), 比如(2,3)这就是(1,2,3)
     这样就统一了：维度为1的会自动广播扩充到另一维度。
"""
# 例如
import numpy as np
from numpy import random
a1 = random.randn(8,1,6,1)
a2 = random.randn(7,1,5)

a3 = a1 + a2 # 根据广播机制原则，从后往前维度比较：1-5,6-1,1-7都符合维度兼容的2个条件之一
             # 所以复制扩展成(8,7,6,5) + (7,6,5) -> (8,7,6,5)+(8,7,6,5)->(8,7,6,5)
# 一维广播
x = np.arange(4)    # (4,)     
xx = x.reshape(4,1) # (4,1)     
y = np.ones(5)      # (5,)
z = np.ones((3,4))  # (3,4)
a = 2
x + y  # 报错：最后一位维度不兼容
xx + y # 可行，等效于(4,1)+(5,)->(4,5)+(1,5)->(4,5)+(4,5)->(4,5)
x + z  # 可行，等效于(4,)+(3,4)=(1,4)+(3,4)->(3,4)+(3,4)->(3,4)
a + x  # 标量本质可看成(1,1) + (1,4) = (1,4) = (4,)

# 高维广播
a = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]]) # (2,2,3)
b = np.array([[1,1,1],[5,5,5]])                     # (2,3)
a = np.ones((2,3,4))
b = np.ones((3,4))
c = a + b   # 可行，等效于(2,2,3)+(2,3)->(2,2,3)+(2,2,3)->(2,2,3)
            # 要注意的是最左边的维度从0扩充到2，就相当于该维度右边的所有维度的数据都要复制扩充

# 真实场景1： anchor平移
base_anchors = np.ones((3,4))   # (3,4)
shifts = np.ones((100,4))       # (100,4)
all_anchors = base_anchors + shifts[:, None, :]  # (100,3,4)
flatten_anchors = all_anchors.reshape(-1, 4)           # (300, 4)
          

""" ----------------------------------------------------------------------
Q: Numpy中的高维array中维度为1代表什么含义？
1. 维度相当于中括号，多少维度就有多少组中括号，比如维度=1相当于加一组空的中括号
2. 扩维的目的：通常的array计算都是按位操作，而有的时候需要对高维数据按行/按列/按层操作，此时
   可以采用扩维思想从高维数据提取出行/列/层进行计算，非常方便
   如(m,n)扩维到(m,1,n)就相当于取出1行n列进行循环，(m,n)扩维到(1,m,n)就相当于取出(m,n)的层进行循环(再广播复制该层)
   所以扩维思想常用于跟广播机制一起使用，对高维数据按照某行，或者某列，或者某层进行计算。
   比如(9,4)与(32700,4)的数据，如果希望(9,4)的数据跟(32700,4)中的每一行都相加，
   就需要先把(32799,4)中扩维出来一个(1,4)，也就相当与把(32700,4)的每行数据分离出来进行计算
   
"""
a = np.array([[1,2,3],[4,5,6]]) # (2,3)

# 以下2种写法等效
a1 = a[None,...]                     # (1,2,3)
a2 = np.array([[[1,2,3],[4,5,6]]])   # (1,2,3)

# 以下两种写法等效
a3 = a[:,None,:]                     # (2,1,3)
a4 = np.array([[[1,2,3]],[[4,5,6]]]) # (2,1,3)
a5 = a[:,None]        # 也可省略后边的维度不写


"""-------------------------------------------------------------------------
Q: Numpy中最重要的一个问题：如何理解高维数组，如何操作高维数组？
1. 高维数组的升1维度：相当于提取某行或某列
2. 高维数组的切片：相当于降维提取子集
"""
a = random.randn(3,4)
b = random.randn(200,4)
xmin = a[:,None,:]



# %%
"""Q: Numpy中的meshgrid()是什么功能的函数？
1. 网格坐标：就是对网格中每一个格子进行坐标定位，比如A(x,y)其中x为横坐标，y为纵坐标
   网格坐标的定位往往把所有网格的x坐标放在一个数组中，所有网格的y坐标放在另一个数组中，便于绘图plot(xx,yy)
   这时就需要一个专门函数把所有网格坐标点生成出来。
   
2. xx, yy = meshgrid(x,y)就是生成网格坐标的函数：
    >x,y分别代表横坐标，纵坐标的一维向量(n,)，比如x=np.array([1,2,3])
    >本质上其实就是把x，y的一维坐标做堆叠+升维，x沿行变化方向(axis=0),y沿列变化方向(axis=1)
"""
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3])
y = np.array([1,2])
xx, yy = np.meshgrid(x,y)
plt.scatter(xx,yy)

# 手动实现pytorch版的meshgrid()，也就相当于手动实现numpy版的meshgrid
x = np.array([1,2,3])
y = np.array([1,2])
xx = np.repeat(x[None,:],len(y),axis=0)
yy = np.repeat(y[:,None],len(x),axis=1)




# %% 
"""另一种理解axis的方法
1. 常规理解： axis=0, 行变换的方向， axis=1，列变换的方向
2. 高维理解： axis=i, 每轮计算固定其他维度该维循环，并最终该维坍缩为0  (非常重要且有价值的理解)
"""
import numpy as np

"""
[[[3, 7, 1, 9],
  [4, 4, 3, 5],
  [6, 4, 9, 1]],

  [[6, 7, 4, 3],
   [6, 6, 4, 3],
   [3, 2, 4, 5]]]
"""
random.seed(1)
a1 = np.random.randint(1,10, size=(2,3,4))
np.sum(a1, axis=0)   # 表示在层这个维度循环计算，固定行列，第一层取1个第二层取1个，再进行求和，最终层这个维度坍缩为0，输出维度(3,4)
np.sum(a1, axis=1)  # 表示在行这个维度循环计算，固定层列，第一行1个第二行1个第三行1个，再进行求和，最终该维坍缩为0输出(2,4)

np.sum(a1, axis=(1,2)) # 表示同时在2个维度循环计算，固定层，行列一起变换求和，最终该维坍缩为0输出(2,)

# %%
"""
矩阵mat和数组array的点乘叉乘的差别:
# 总结下来就是： 
    mulitiply为按位(就是普通乘法，每一位单独做，所以用普通的multiply乘法单词表示)，
    dot为矩阵乘法(一行对一列累加缩减为1个值，这是dot的由来)
# 而最常用的*则被运算符重载为不同功能：
    array偏重常规运算所以*重载为按位运算，也就是*=np.multiply 
    mat偏重矩阵运算所以*重载为矩阵乘法,也就是*=np.dot
# mat还有一个跟array最大的不同：
    mat作为矩阵，没有一维这种(n,)类型，所以切片以后也是(1,n), 比如mat_a为(2,2)， 切片后为(1,2)
    array作为数组，切片会降维，比如array_a为(2,2)，切片后为(2,)，实际是省略了(1,2)中的1，但在广播时会自动恢复。
"""
import numpy as np

ar1 = np.array([[1,2],
                [2,1]])
ar2 = np.array([[1,0],
                [1,0]])

mt1 = np.mat([[1,2],
              [2,1]])
mt2 = np.mat([[1,0],
              [1,0]])

#----------------数组array----------------
ar1*ar2                       # * 代表按位
np.multiply(ar1,ar2)          # multiply代表按位
np.dot(ar1,ar2)               # dot代表矩阵乘法

#----------------矩阵mat----------------
mt1*mt2                       # * 代表矩阵乘法
np.multiply(mt1, mt2)         # muliply 代表按位
np.dot(mt1,mt2)               # 代表矩阵乘法

# %%
"""Matplotlib.pyplot的几个核心绘图命令
在spyder中设置绘图单独窗口显示：
Tools > Preferences > IPython Console > Graphics > Graphics backend, inline 即终端输出，Qt则是新窗口输出
"""
import matplotlib.pyplot as plt
import numpy as np



# 线条图plt.plot----------------------------
# 实例参考如下一个经典的各种设置都有的案例
# https://mp.weixin.qq.com/s?__biz=MzIxNjY2MTQ2NA==&mid=2247484018&idx=1&sn=ca15d8b224e58d85668e2ddcf3b4f424&chksm=9784eec5a0f367d36be1ff009247ef1b49bf4b479a03d6fdf68f8aba2c0ee487bf71a46cfd80#rd
x = np.arange(-10,10,1)
y = x * x
color = [[0,0,1],[0,1,0]]  # 可以输入字符'r'/'g'/'b'，也可以输入取值(0-1)之间的list代表rgb颜色表，也可以输入array。
plt.figure()    # 创建一个新图
m1 = plt.plot(x, y, color=color[0], marker='o',    # marker可以为'o', ''
              linewidth=1.0, linestyle='--',    # 线宽和线型
              label='y=x^2')  # 如果有设置label，则只要plt.legend()即可显示 
                                                                           
m2 = plt.scatter(x, y+10, color='blue', label='y=x^2+10')
plt.title('show the curve result')

#plt.xlabel('X')
#plt.ylabel('Y')
plt.xlim((-8,8))  # 设置x轴的显示范围
plt.ylim((0,20))  # 设置y轴的显示范围

ticks = np.linspace(-4,4,9)  #刻度范围(-1,2),共计5个
plt.xticks(ticks)            # 设置x轴刻度线显示内容
plt.yticks([-2.2, -1, 1, 1.5, 2.4],
           ['really bad','bad','normal','good','really good']) # 设置y轴刻度线显示内容
plt.grid()  # 设置网格(只会显示ticks设置的点的网格)
plt.legend(loc = 'upper right')      # 设置图例位置：'upper right', 'lower left', 'best'
#plt.legend(handles=[m1, m2], labels=['1', '2'], loc='upper right') # 也可以这样设置图例，则绘制时不需要指定label

ax = plt.gca()  # 获得坐标轴
ax.spines['right'].set_color('none')  # 设置坐标轴颜色：'left' 'right' 'top' 'bottom'
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')    # 设置x轴刻度显示的位置：可以显示在top, bottom, both, default, none
ax.spines['bottom'].set_position(('data', 0))  # 设置bottom边框的位置在数据的y=0的位置, data代表该边框在数据中，也可以为'data' 'outward' 'axes'
ax.yaxis.set_ticks_position('left')    # 设置x轴刻度显示的位置：可以显示在left, right, both, default, none
ax.spines['left'].set_position(('outward', 0))  # 设置left边框的位置在x=0的位置，
#ax.set_title('ax title', fontsize=14, color='r')  # 这个title跟plt.title()只有一个会显示，另一个会被覆盖


# 散点图plt.scatter--------------------------
colors = np.array([[0.0, 0.0, 0.0], [0.5019607843137255, 0.0, 0.0], [0.0, 0.5019607843137255, 0.0]])
n=3   #数据大小
X=np.random.normal(0,1,n)   #每一个点的X值
Y=np.random.normal(0,1,n)   #
T=np.arctan2(Y,X)
plt.scatter(X,Y,s=75,c=colors,alpha=.5)


# 柱状图plt.bar------------------------------
n=12
X=np.arange(n)
Y1=(1 - X / float(n)) * np.random.uniform(0.5,1.0,n)
Y2=(1 - X / float(n)) * np.random.uniform(0.5,1.0,n)
plt.bar(X,+Y1)
plt.bar(X,-Y2)
plt.xlim(-1,n)
plt.xticks(())
plt.ylim(-1.25,1.25)
plt.yticks(())
plt.bar(X,+Y1,facecolor='#9999ff',edgecolor='white')   # 设置主题颜色和边框颜色
plt.bar(X,-Y2,facecolor='#ff9999',edgecolor='white')
for x,y in zip(X,Y1):
    plt.text(x+0.4,y+0.05,'%.2f' % y,ha='center',va='bottom')  # 增加显示数值
for x,y in zip(X,Y2):
    plt.text(x+0.4,-y-0.05,'%.2f' % y,ha='center',va='bottom')

# 绘制多图----------------------------------
plt.subplot(131), plt.imshow(img1), plt.title('img1')
plt.subplot(132), plt.imshow(img2), plt.title('img2')
plt.subplot(133), plt.imshow(img3), plt.title('img3')
    
# 等高线图plt.contour, plt.contourf------------------------
def f(x,y):
    return (1 - x/2 + x**5 + y**3) * np.exp(-x**2,-y**2)
n=256
x=np.linspace(-3,3,n)
y=np.linspace(-3,3,n)
X,Y=np.meshgrid(x,y)    # (m,n)
Z=f(X,Y)                # (m,n)
plt.contourf(X,Y,Z,8,alpha=0.75,cmap=plt.cm.hot)      # 填充每一部分的颜色 (X, Y, Z)
                                                      # X,Y,Z为相同shape的数组，Z代表高度   
C=plt.contour(X,Y,Z,8,colors='black',linewidth=0.5)   # 绘制每一部分之间的边界 

# 绘制直方图: data需要是一个展平的list, bins为柱子个数，range为个数范围显示(即x坐标的个数统计范围)
# alpha为透明度，facecolor为柱子颜色
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('test/test_data/messi.jpg')
b,g,r = cv2.split(img)
plt.hist(b.flatten(), bins=256, range=[0,256], alpha= 0.5, facecolor='blue')
plt.hist(g.flatten(), bins=256, range=[200,256], alpha= 0.5, facecolor='green')
plt.hist(r.flatten(), bins=256, range=[0,256], alpha= 0.5, facecolor='red')







# %%
"""pandas的基本操作
"""
import pandas as pd
# 用pandas的最常用地方是读取数据为dataframe格式
data = pd.read_csv(path)

# 转换df格式为numpy
data1 = np.array(data)

# df创建

# df切片
r1 = data.loc[0]  # 提取第0行
r2 = data.loc[1]
d0 = data[0]     # 提取某列
d1 = data[1]


# df的常用函数
data.sum()
data.mean()
data.var()
data.std()

data.corr(method='pearson')    # 获得整个相关系数矩阵
r1.corr(r2, method='pearson')  # 获得两行之间的相关系数

data.cov()  # 获得整个协方差矩阵
d0.cov(d1)  # 获得两列之间的相关系数

data.describe()  # 获得基本统计量计算结果

