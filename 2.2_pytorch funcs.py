#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 09:49:15 2019

@author: suliang
"""


'''-----------------------------data-------------------------------------------
Q. tensor的创建
1. pytorch中数据类型：其中最常用的torch.float/torch.long
    dtype=torch.float32/torch.float, 代表float32 (常用，是pytorch要求的输入类型)
    dtype=torch.float64/torch.double, 代表float64
    dtype=torch.uint8, 代表uint8
    dtype=torch.int8, 代表int8,
    dtype=torch.int16/torch.short, 代表int16,
    dtype=torch.int32/torch.int, 代表int32,
    dtype=torch.int64/torch.long, 代表int64 (常用，是)
2. 
'''
import torch
f1 = torch.tensor([[1,2,3],[4,5,6]])  # 默认float
f2 = torch.tensor(2,3)
f3 = torch.IntTensor([1,2,3])  # 整数tensor

torch.ones(2,3)
torch.zeros(2,3)
torch.eye(2,3)
torch.empty(2,3)
torch.full((2,3),10)
torch.arange(1,10,2)  # (start, end, step)这跟切片一样，跟array统一
torch.linspace(3,10,5) # (start, end, n_num)

# 另一种是对照一个已有tensor创建一个新tensor,具有相同dtype/device
# 而size可以重新指定，填充value也可指定
t1 = torch.tensor([[1.5, 2.1],[3.2, 4.7]])
t1.new_zeros((2,3), dtype=torch.float32)    # 填充0
t1.new_ones((2,3), dtype=torch.float32)     # 填充1
t1.new_full((2,3),3.5, dtype=torch.float32) # 填充任意值
t1.new_empty((2,3), dtype=torch.float32)

t0 = torch.tensor([[1,2,3],[3,2,1]])
torch.zeros_like(t0)      # 填充0
torch.ones_like(t0)       # 填充1
torch.empty_like(t0)      # 填充随机值
torch.full_like(t0, 3.5)  # 填充任意值，dtype会强制转换到源数据的格式，也可用dtype指定
torch.rand_like(t0, dtype=torch.float32)   # 填充平均分布[0,1]的随机值, 这里需要指定dtype否则沿用了t0的dtype，就会报错
torch.randint_like(t0, 1,10)
torch.randn_like(t0, dtype=torch.float32)

# 还有一种创建tensor的体系：先创建空的指定类型的tensor,然后初始化
t1 = torch.tensor((), dtype=torch.float32)
t1 = torch.FloatTensor().new_zeros((2,3))
t1 = torch.IntTensor().new_full((2,3))
t1 = torch.LongTensor().new_full((2,3))

# 创建时指定数据格式和是否求导
t0 = torch.tensor([1.,2.,3.], requires_grad=True)     # 只要带小数点就是float
t1 = torch.tensor(np.array([1,2,3], dtype=np.float32), requires_grad=True) # 在np中指定数据格式


'''------------------------------------------------------------------------
Q. 创建tensor的几种方法的差异？
1. torch.tensor(ndarray) 为深拷贝，推荐使用
2. torch.from_numpy(ndarray) 为浅拷贝
'''
import numpy as np
data = np.array([1,2,3])
t1 = torch.tensor(data)     # 属于深拷贝数据，不会随data而变，他等效于x.clone().detach()
t2 = torch.from_numpy(data) # 属于浅拷贝，会随data而变
data += 1
print(data, t1,t2)


'''-----------------------------------------------------------------------
Q. pytorch如何产生随机数？
1. 生成随机数: torch.rand(), torch.randn(), torch.randint()

2. 随机乱序: torch.randperm(int)

3. 随机抽样：torch没有这种函数，只能用torch.randperm(int)结合index来实现
'''
# 选分布，定义size: 取值范围只在(0,1) (对应np.random.rand(), np.random.randn())
torch.rand(2,3)                  # 实数，指定均匀分布(0,1)，定义尺寸
torch.randn(2,3)                 # 实数，指定标准正态分布N(0,1)，定义尺寸

# 选取值范围，定义size：分布只为均匀分布 (对应np.random.randint(), np.random.uniform())
torch.randint(1,10, size=(2,3))  # 整数，指定均匀分布(low, high)，定义取值范围和尺寸
                                 # 似乎缺少一个numpy的uniform: 实数版
b4 = np.random.uniform(1,10,size=(2,3))

# 随机抽取(对应numpy的np.random.choic(lst))： pytorch没有，可间接用torch.randperm实现

# 随机乱序(对应numpy的np.random.permutation(), np.random.shuffle())
# torch中做随机抽样的函数只有torch.randperm()结合index来做
# torch.randperm(int) 可以在int处输入index的len，即可对index进行打乱，然后抽样即可
t = torch.tensor([30,10,55,73,42,21,93,81,32])
rand_ind_3 = torch.randperm(len(t))[:3]        # 先把index随机，并提取出需要的随机个数
t1 = t[rand_ind_3]                             # 切片得到随机值

'''-----------------------------------------------------------------------
Q. tensor的核心计算函数？
pytorch对tensor的操作都是基于元素进行操作
torch.clamp(low, high), torch.exp(ti), torch.pow(ti, value), torch.mul(ti, value),
torch.log(ti), torch.log10(ti), 
'''
t0 = torch.tensor([-1.2, 0, 1.5])
# clamp
torch.clamp(t0, -0.5,0.5)
# exp
t1 = torch.exp(t0)
# log()和log10()
t2 = torch.log(t1)
t3 = torch.log10(t1)
t4 = torch.log2(t1)
# abs
t3 = torch.abs(t0)
# add (也可用加法重载运算符)
t4 = torch.add(t0, 100)
t0 + 100
# mul (也可用乘法重载运算符)
t7 = torch.mul(t0, 100)
t0 * 100
# ceil, floor
t5 = torch.ceil(t0)
t6 = torch.floor(t0)
t8 = torch.round(t0)
# pow
t8 = torch.pow(t0, 2)
# sigmoid()
t9 = torch.sigmoid(t0)
# sign
t10 = torch.sign(t0)
# sqrt
t11 = torch.sqrt(t1)
# sin, cos, tanh, tan,
t12 = torch.sin(t1)


'''-----------------------------------------------------------------------
Q. tensor的reduction缩减（规约）计算函数有哪些？
缩减操作的函数不多：
1. max/min/argmax/argmin: 不带dim则返回一个值，带dim则返回tuple(tensor(max),tensor(argmax))
   注意pytorch里边max()函数非常特殊，在指定dim可以同时返回value和index
2. sum/cumsum
3. mean/std/var/median/mode: 只有这组不能带dim
'''
t0 = torch.tensor([[-1.0, 0, 3., 2.],[1.5, 3, -4., 0.8]])
# max, min
t2 = t0.max() # 不带dim, 返回一个值
t3 = t0.min()  
t4 = t0.max(dim=0) # 带dim，返回两个tensor,一个是max,一个是index
t1 = torch.argmax(t0, dim=0)  # 返回的是该direction方向index
# sum
t4 = torch.sum(t0)
t5 = torch.sum(t0, dim=1)
# cumsum
t5 = torch.cumsum(t0,dim=0)  # 逐步累加
# mean, std, var, median, mode
t6 = t0.mean()
t6-1 = t0.mean()
t7 = t0.std()
t8 = t0.median()
t9 = t0.mode()
t10 = t0.var()


'''-----------------------------------------------------------------------
Q. tensor的其他规约操作scatter, gather函数有哪些，有什么功能，如何使用？
1. scatter_(1,data,1)可用于转换数据为one-hot-code独热编码
2. gather()
'''
data = torch.LongTensor([1,0,4])
one_hot_code = torch.zeros(3,5).scatter_(1, data.view(-1,1), 1)



'''-----------------------------------------------------------------------
Q. tensor的排序和筛选
一个基本思想是：筛选一般都是返回bool，所以numpy/torch基本也是这个原则

1. 首先参考numpy:
(1)numpy排序: 
    排序返回数值: a.sort()在原数据上直接操作, sorted(a)返回副本原数据不变
    排序返回index: np.argsort(a)
(2)numpy筛选：
    筛选返回bool: a>0
    筛选返回value: a[a>0]
    筛选返回index: np.where(a>0)
2. pytorch的实施
(1) tensor的排序
    排序返回数值/index: t1, indics = torch.sort(t, dim=1)
    排序返回index: torch.argsort()
(2) tensor的筛选(2类，一类返回bool，一类返回index)：
    筛选返回bool： 有如下函数，也有重载运算符
        torch.gt(t, value) ，大于
        torch.ge(t, value)，大于等于
        torch.eq(t, value), 等于，还有一个torch.equal(t1,t2)是比较函数而不是筛选
        torch.lt(t, value)，小于
        torch.le(t, value)，小于等于
        torch.ne(t), 不等于
    筛选返回index/value:
        torch.nonzero(t): 筛选返回非0的index
        torch.topk(t, k, dim=1): 筛选返回topk的值和index, 返回的是最大的k个，也可设置largest=False返回最小的k个
        torch.max(t, dim=1): 筛选返回max的值和index
'''
# -------tensor排序---------------
t = torch.tensor([[2,0,13,-8,0],[-3,7,0,32,3]])
t.sort(dim=0)
# -------tensor筛选---------------
t = torch.tensor([[2,0,13,-8,0],[-3,7,0,32,3]])
# 获得bool
t > 0
t >=13
t ==0
t != 0
# 获得index
torch.nonzero(t)
torch.topk(t,3,dim=1)


'''---------------------------------------------------------------------
Q. torch的截断操作clamp
'''
t = torch.tensor([[1,2,3],[100,200,300]])
torch.clamp(t, min=0., max=10.)     # 目标数据是int, 则截断参数也是int (但实际上不会报错)

t1 = torch.tensor([[1.3,2.5,3.6],[10.1,20.2,30.5]])
torch.clamp(t1, min=2, max=10)


'''-----------------------------------------------------------------------
Q. tensor的比较函数有哪些？
'''
# equal, ge, gt, le, lt, ne 除了
torch.equal(t0,t1)
torch.ge(t0,t1)
torch.gt(t0,t1)
torch.ne(t0,t1)
# 特殊取值判断
torch.isfinite(t2)
torch.isinf(t3)
torch.isnan(t4)
# 排序
torch.sort(t0,dim=0)
torch.argsort(t0,dim=0)
torch.topk(t1, 3, dim=1)


'''------------------------------------------------------------------------
Q. tensor的数据格式转换
'''
import torch
import numpy as np

t1 = torch.tensor([5])
# 修改数据格式
t1 = t1.float()         # 注意.float()操作不是in place操作，且默认是float32
# 修改requires_grad
t1.requires_grad=True  # 注意requires_grad的前提是数据为float格式

# 转成tensor
a = [1,2,3]
b = np.array([3,2,1])
c = dict(a=1,b=2)
torch.tensor(a)  # list转tensor
torch.tensor(b)  # array转tensor
torch.tensor(c)  # dict不可以转tensor

# tensor转其他
t1 = torch.tensor([1,2,3])
t2 = t1.numpy()             # tensor转numpy
t3 = t1.numpy().tolist()    # tensor转numpy,numpy再转list

b0 = torch.tensor(3)
b1 = b0.item()              # 单tensor转标量


'''------------------------------------------------------------------------
Q. tensor的维度转换
1. 维度变换：t.transpose(), t.permute()
2. 维度加减：t.view(), t.reshape(), t.squeeze(), t.unsqueeze()
   也可针对tensor使用numpy认可的None的方式 t[None, :], t[:, None], t[..., None]
'''
from numpy import random
import torch
d0 = random.randint(0,255, size=(300,500))
t0 = random.randint(0,255, size=(3,300,500)) # (c,h,w)

# 维度变换顺序
d1 = torch.tensor(d0) 
d2 = d1.transpose(1,0)  # transpose()只能用于2维

t1 = torch.tensor(t0)  # (c,h,w)
t2 = t1.permute(1,2,0)  # (h,w,c), permute()可以用于更多维

# 行或者列变换顺序

# 维度增减
b0 = torch.tensor([[1,2,3,4,5,6],[4,5,6,7,8,9]])  # (2,6)
b1 = b0.unsqueeze(0)                  # (1,2,6)     
b2 = b1.squeeze(0)

b3 = b0.unsqueeze(1)
b4 = b3.squeeze(1)

b5 = b0.reshape(1,-1)  # (1, n)，相当于展平
b6 = b0.reshape(6,-1)

b7 = b0.view(-1,1)     # (n, 1)  view跟reshape功能一样，
                       # 但view是专供tensor，reshape适用范围更广，且reshape不怕有contiguous问题的数据

b8 = b0.view(1,-1)    #变为二维数组(1,n)
b9 = b0.view(-1)      #变为一维数组(n,)  这个用法也很常见，很方便————————最常用多维变一维

c1 = torch.tensor([1,2,3,4,5])      # 一维tensor(5,)
c2 = c1[:,None]                     # 升维到二维tensor(5,1)————————最常用一维变多维



'''-------------------------------------------------------------------------
Q. 上一问题中发现有很多不同的命令实现展平，但展平算法一样吗？
结论：展平算法都一样，从第一个维度循环，然后到第二个维度
1. 二维展平：逐行展平
3. 三维展平：先到层，第0层逐行展平，第1行逐行展平....
'''
import torch
# 二维展平
a = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]])  
a.flatten()
a.view(-1)
a.reshape(-1)
a.view(1,-1)
a.view(-1,1)

# 三维展平
a = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
a.flatten()
a.view(-1)
a.reshape(-1,1)


'''-----------------------------------------------------------------------
Q. tensor的多维相加怎么做？
1. 扩维相加：这种扩维相加是继承自numpy，适合于高维数组的循环运算
'''
t1 = torch.tensor([[1,2],[3,4],[5,6],[7,8]])  # (4,2)
t2 = torch.tensor([[1,2],[3,4],[5,6]]) # (3,2)

t3 = t1[None, :, :] 
t4 = t2[:, None, :]

'''------------------------------------------------------------------------
Q. tensor的转置跟python不太一样，如何使用，如何避免not contiguous的问题？
1. python 用transpose(m,n,l)可以对3d进行转置，但tensor的transpose(a,b)只能转置2d
   要转置3d需要用permute()
2. 不连续问题解决办法：由于pytorch的transpose/permute会导致不连续的问题，解决方案如下
    >data.contiguous()函数
    >data[...,[2,1,0]]切片运算
    >reshape()函数
'''
import torch
from numpy import random
# not contiguous的问题：来自pytorch的transpose/permute函数，用切片代替就不会有问题
# 解决方案1：a.contiguous()函数
# 解决方案2：用切片替代transpose/permute
# 解决方案3：用reshape替代view
a0 = torch.tensor(random.randint(1,10,size=(10,10,3)))
a1 = a0.permute(2,0,1)
a1.is_contiguous()  # permute后不连续
a1.view(10,5,6)     # 因为not contiguous报错
a1.contiguous().is_contiguous()
a1.contiguous().view(10,5,6)   # 解决(contiguous函数)
a1.reshape(10,5,6)     # 解决(reshape替代view)

a2 = a0[...,[2,0,1]]   # 解决(用切片代替transpose/permute)
a2.is_contiguous()  # 用切片后是连续的
a3 = a0[...,::-1]   # tensor还不支持负的step
a3.is_contiguous()  # 该操作还不能判断

b0 = torch.tensor(random.randint(1,10,size=(3,4)))
b1 = b0.transpose(1,0)
b1.is_contiguous()  # transpose后不连续


'''-------------------------------------------------------------------------
Q. 对tensor的堆叠
1. python堆叠用np.stack(), np.concatenate()
2. tensor堆叠用torch.stack(), torch.cat()
3. 都是1维堆叠用stack (升维堆叠成2维), 二维堆叠用concatenate (维持二维堆叠)
'''
t1 = torch.tensor([1,2,3,4,5])
t2 = torch.tensor([6,7,8,9,10])
torch.stack((t1,t2),0)  # 注意 torch使用dim关键字代替了numpy的axis
torch.stack((t1,t2),1)  #
# stack([t1,t2],-1) dim=-1的用法
torch.stack([t1,t2],-1) # dim=-1的理解参考np.stack()非常顺的去理解(m,)升维到(m,1)然后n个堆叠成(m,n)，顺滑！
                        # 在一些算法中dim=-1是经常被使用的一个小方法。

# torch.stack()天然跟list融合，因为list一般都是一维的
# 用list作为存放各种数据的容器，然后把这个list往stack里一丢，非常方便 
lst = [t1,t2,t1,t2]
torch.stack(lst, dim=1)
                        
# torch.cat()                        
t4 = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]])
t5 = torch.tensor([[11,12,13,14,15],[16,17,18,19,20]])
t6 = torch.cat((t4,t5),0)
t7 = torch.cat((t4,t5),1)

# 对比numpy的堆叠
import numpy as np
d1 = [1,2,3,4,5]
d2 = [6,7,8,9,10]
np.stack((d1,d2),0)  # numpy使用axis关键字
np.stack([d1,d2],1)

d4 = [[1,2,3,4,5],[6,7,8,9,10]]
d5 = [[11,12,13,14,15],[16,17,18,19,20]]
d7 = np.concatenate((d4,d5),0)
d8 = np.concatenate((d4,d5),1)


'''-------------------------------------------------------------------------
Q. 对tensor的广播式扩展，跟堆叠有什么区别？
1. repeat是把同一个数据堆叠，而stack/cat是把不同数据堆叠
2. t.repeat(m,n)是把原数据堆叠成m行，n列，这是更便捷的行列同时堆叠，而stack/cat是一次只能往一个方向堆叠

# 注意跟numpy的区别：
1. torch的repeat是tensor的属性，只能后置，不能前置函数式
2. torch的repeat的参数是(m,n)m行n列，可同时多维堆叠，其实相当于np.tile()函数 (np.repeat是单个方向堆叠，np.tile()是多个方向堆叠)
   nunmpy的repeat只能堆叠一个维度方向（且一维只能水平水平，一维竖直堆叠就需要先扩维到2维再完成)
'''
t1 = torch.tensor([1,2,3,4,5])
t2 = torch.tensor([6,7,8,9,10])
t3 = torch.stack((t1,t2),0)  # stack在行方向上堆叠

# t.repeat()相当与tile,非常方便
t1.repeat(3,2)

# 另一种同一数据的堆叠方法是用t.expand(),该方法跟t.repeat()类似,但只适合二维
# 并且expand的行列计数方式跟repeat不同，他采用的是定义生成的结果数组的行数和列数
# 而repeat是定义把源数组看成一个元素堆叠成几行几列。
t1 = torch.tensor([1,2,3,4,5])
t1.expand(2,5)          # 堆叠成2行5列元素
t1.repeat(2,1)          # 等效写法：源数组整体堆叠成2行1列

t1[:,None].expand(5,3)  # 变列，堆叠成5行3列个元素
t1[:,None].expand(-1,3) # 等效写法，不变的元素维度省略成-1
t1[:,None].repeat(1,3)   # 等效写法，原数组整体堆叠成1行3列

t2 = torch.tensor([[1,2,3],[4,5,6]])
t2.expand(2,6)

# 跟numpy的区别：torch的repeat跟np.repeat不太一样，反而类似于np.tile()
d1 = np.array([1,2,3,4,5])
d2 = np.repeat(d1,3,axis=0)      # 只能这样水平堆叠，如果设置axis=1的堆叠就会报错
d3 = np.repeat(d1[None, :],3,axis=0)  # np.repeat()单维度堆叠

d4 = np.tile(d1, (3,3))          # 多维度堆叠

t1 = torch.tensor([1,2,3,4,5])   
t4 = t1.repeat((3,3))            # torch的repeat()结果跟np.tile()一样，都是可以同时在多个维度同时堆叠


'''-----------------------------------------------------------------------
Q. 对tensor的展平？
1. torch.flatten() 跟np.flatten()基本一致（flatten都是非inplace操作）
2. torch没有类似于np的ravel()的inplace操作函数
'''
t1 = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]])
t2 = t1.flatten()  # 此时t1不变



'''----------------------------------------------------------------------
Q. 对于修改tensor需要注意的问题？
参考：https://zhuanlan.zhihu.com/p/38475183
1.有两种情况in place修改tensor是会产生错误的，需要注意避免(其中一种情况可能不会报错)：
    情况1: 需要梯度求导的叶子节点被修改时会报错，需要用.data直接修改，或者detach到另一个变量修改(但属于浅复制)  
    比如t.numpy()报错，需要先t.detach().numpy()
    比如t+=1报错，需要先t.data+=1
    情况2: 需要梯度求导的中间节点被修改时会报错，需要用.data直接修改，或者detach到另一个变量修改(但属于浅复制)
2. 区别t.data和t.detach()：
   t.detach()操作是把tensor从计算图解放出来，会把requires_grad设置为false，就可以操作了
   t.data操作返回的tensor也是requires_grad为false，也就可以操作了
   注意t.detach()与t.data的区别：t.data的修改虽然影响了计算图，但系统不报错导致用户不知道，所以不建议使用
   尽可能用detach()函数，他在例子1/2两种情况都会主动报错，而不会计算出隐性错误的结果。
'''
import torch
import numpy as np
a0 = np.array([[1,2,3],[4,5,6]],dtype=np.float32)
a1 = torch.tensor(a0, requires_grad=True)

# 例子1: 叶子节点如果需要自动求导，就会被引用和预存储，所以不能in place修改
a1 += 1         # 报错：因为a1是requires_grad=True, 不能直接修改
b1 = a1.numpy() # 报错：因为a1是requries_grad=True, 不能直接.numpy()

a1.data += 1             # 修改方式：可以考虑取data后修改，这也是参数w初始化的方法
b1 = a1.detach().numpy() # 修改方式：可以考虑先detach()跟计算图分离，然后再转numpy()

# 例子2: 中间变量d1已纳入计算图，在自动求导之前不能in place修改，否则报错
x = torch.tensor([1.,2.], requires_grad=True)
y = torch.tensor([[2.],[1.]], requires_grad=True)
z = torch.tensor([3.], requires_grad=True)
d1 = torch.matmul(x,y)
d2 = torch.matmul(d1,z)
d1_ = d1.detach()   # 此处用detach()分离出d1_，但依然指向同一内存，所以如果修改理论上是影响了计算图，所以会报错
d1_[:] = 1          # 而如果用.data分离出d1_，即使修改也不会报错，导致用户无法察觉的错误，更严重，所以不建议用.data,而建议用.detach()
d2.backward()       # 报错



'''----------------------------------------------------------------------
Q. tensor的几个核心属性如何使用？
1. t.data 返回的是一个tensor,该tensor为requires_grad=False，相当于把tensor从计算图分离的tensor
2. t.grad 返回的是一个tensor,该tensor同样requires_grad=False
   t.grad.data 返回的是一个tensor,该tensor同样requires_grad=False
3. 梯度清零的2种方式：t.grad.data.zero_(), optimizer.zero_grad()
'''
import numpy as np
import torch
d1 = torch.tensor(np.array([1.]),requires_grad=True)
d2 = d1.data   # 此tensor的requires_grad=false
d3 = d1.grad   # 初始梯度=None

# 反向传播之后
d1.backward()     
d6 = d1.grad            # 反向传播后返回一个tensor, false
d7 = d1.grad.data       # t.data操作可让grad这个tensor跟计算图分离，从而可以进行修改，比如梯度清零
d1.grad.data.zero_()    # 梯度清零, 这里我理解d1.grad.zero_()跟d1.grad.data.zero_()是一样的，因为d1.grad返回的tensor没有requires_grad=True的问题。



'''----------------------------------------------------------------------
Q. 对tensor类型的定义？
1. conv层只接受np.float32类型，如果不是会报错
2. conv层只有3个位置参数是必须输入的：in/out/ker，其他都是关键字参数s/p/d/b用默认值也可以1/0/1/true
'''
# conv层的计算输入
import torch
import torch.nn as nn
from numpy import random
d1 = torch.tensor(random.uniform(-1.5,1.5, size=(8, 3, 300, 240)).astype(np.float32))  # 自定义数据需要转换成np.float32
conv = nn.Conv2d(3, 64, 3, stride=1, padding=0, dilation=1, bias=True)  # 除了in/out/ker之外，剩下都是关键字参数可不写则用默认参数1/0/1/True
out = conv(d1)


'''------------------------------------------------------------------------
Q. 对checkpoint/state_dict的加载与保存操作如何进行？
'''

# %%
'''-------------------------------module----------------------------------
Q. 在深入各个子模型之前，如何跑一个最小系统？
'''


# %%
'''-------------------------------module----------------------------------
Q.在pytorch中model的本质是什么，有哪几种model
1. 核心概念：所有layer/model核心都是nn.module的继承，module基类包含了
2. 基类module的核心属性：
    model._buffers，为OrderedDict变量，以下为相关函数：
        对应model.buffers(), 为iterator，存放所有buffers
        对应model.named_buffers(): 为iterator，存放带名称buffers
    model._parameters，为(OrderedDict)，以下为相关函数：
        对应model.parameters(): 为iterator，存放所有模型参数
        对应model.named_parameters(): 为iterator，存放带名称模型参数
    model._modules，为(OrderedDict)，以下为相关函数：
        对应model.named_modules，为(iterator)，是返回了所有有名称的module，包含了主模型/子模型/层
        对应model.modules，为(iterator)，从named_modules得到生成器
        还有model.named_chilren()，为iterator，是返回了子模型module
        还有model.children()，从named_children得到生成器

3. 基类module的重要方法：
    >model.add_module(name, new_model)用来添加子模型，等小于添加一个module的属性，比如model.sub_module_name = new_module
        所以self.add_module(name, new_model)等价于self.name = new_model
        可用来添加子模型
    >model.apply(fn)用来对每个子module实施fn,
        可用来init_weight
    >model.cuda(device=fn)用来调用_apply()把所有模型/子模型的参数都实施该fn(param.data)，
        可用来把模型参数传入device
    >model.load_state_dict() 用来加载已有模型的所有参数到本模型
        可用来导入预训练参数
    >model.train() 用于把module以及子模型的self.training标志位设置为True
        可用来实施training的指示
    >model.zero_grad() 用于把self.parameters里边所有参数的grad都设置为0
        可用来初始化梯度grad

3.module的核心运行逻辑
    >创建新的子模型时：conv1=nn.Conv2d(2,2,3), 则调用__setattr__更新_modules,_parameters,_buffers
    >
'''
# ------level1: module基类------
class Module(object):
    dump_patches = False
    _version = 1
    def __init__(self):
        self._parameters = OrderedDict() # 存放每个层的参数
        self._buffers = OrderedDict()    # 存放中间计算变量
        self._modules = OrderedDict()    # 存放子模型
        continue
    def forward(self, *input):
        """这个方法用于主要的前向计算,是__call__方法实际调用的函数"""
        raise NotImplementedError
    def register_buffer(self, name, tensor):
        self._buffers[name] = tensor
        continue
    def register_parameter(self, name, param):
        self._parameters[name] = param
        continue
    def add_module(self, name, module):
        self._modules[name] = module
        continue
    def _apply(self, fn):
        """用于传入一个函数fn，实施到所有子module，同时实施到_parameters/_buffers"""
        for module in self.children():
            module._apply(fn)
        for param in self._parameters.values():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        return self
    def apply(self, fn):
        """用于传入一个函数fn，实施到所有子module上，该fn的参数输入必须是module"""
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self
    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))
    def cpu(self):
        return self._apply(lambda t: t.cpu())
    def type(self, dst_type):
        return self._apply(lambda t: t.type(dst_type))
    def float(self):
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)
    def double(self):
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)
    def half(self):
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)
    def to(self, *args, **kwargs):
        continue
    def register_backward_hook(self, hook):
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle
    def register_forward_pre_hook(self, hook):
        handle = hooks.RemovableHandle(self._forward_pre_hooks)
        self._forward_pre_hooks[handle.id] = hook
        return handle
    def register_forward_hook(self, hook):
        handle = hooks.RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle
    def _tracing_name(self, tracing_state):
        continue
    def _slow_forward(self, *input, **kwargs):
        continue
    def __call__(self, *input, **kwargs):
        """这是model在运行的核心过程："""
        for hook in self._forward_pre_hooks.values():  # 先运行_forward_pre_hooks里边的hook
            hook(self,input)
        if torch._C._get_tracing_state():  # 这个应该是调试用的
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(*input, **kwargs)   # 计算模型输出
        for hook in self._forward_hooks.values():     # 再运行_forward_hooks里边的hook
            hook_result = hook()
        if len(self._backward_hooks)>0:               # 再运行_backward_hooks
            var = result                              
            while not isinstance(var, torch.Tensor):  
                if isinstance(var, dict):
                    var= next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)  # 把backward hook函数先绑定一个self形参
                    functools.update_wrapper(wrapper, hook)  # 更新hook的相关原始属性给wrapper
                    grad_fn.register_hook(wrapper)
        return result
    def __setstate__(self, state):
        continue
    def __getattr__(self, name):
        continue
    def __setattr__(self, name, value):
        """定义了setattr方法，所以module.conv1=nn.conv2d()才能实现
        每次创建新的子模型，就会更新_parameters/
        同时add_module()能把module注册进_modules，而model.module_name = moduel也
        因为__setattr__的定义能够把module注册进_modules里去，两者等效
        """
        self.register_parameter(name, value)
        modules[name] = value
        buffers[name] = value
        continue
    def __delattr__(self, name):
        continue
    def _register_state_dict_hook(self, hook):
        continue
    def _register_load_state_dict_pre_hook(self, hook):
        continue
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        continue
    def load_state_dict(self, state_dict, strict=True):
        continue
    def load(module, prefix=''):
        continue
    def parameters(self, recurse=True):
        """从named_parameters获得数据创建生成器"""
        for name, param in self.named_parameters(recurse=recurse):
            yield param
    def named_parameters(self, prefix='', recurse=True):
        """创建生成器"""
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem
    def buffers(self, recurse=True):
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf
    def named_buffers(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem
    def children(self):
        for name, module in self.named_children():
            yield module
    def named_children(self):
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module
    def modules(self):
        for name, module in self.named_modules():
            yield module
    def named_modules(self, memo=None, prefix=''):
        continue
    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
    def eval(self):
        return self.train(False)
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    def share_memory(self):
        return self._apply(lambda t: t.share_memory_())
    def _get_name(self):
        return self.__class__.__name__
    def __repr__(self):
        continue
    def __dir__(self):
        continue

# ------level2: layers继承Module的子类------
class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        continue
    def reset_parameters(self):
        """初始化module的参数"""
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
# ------level3: layers继承Module的子类------            
class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        continue
    def forward():
        """"""
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
# ------level4: layers函数------  
import torch.nn.functional as F
F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
"""对输入的多层图片数据进行卷积计算，返回tensor"""
# ------level5: Sequential类/ModuleList类/ModuleDict类------ 
class Sequential(Module):
    """作为modules子类，额外实现setitem/getitem作为切片手段,实现类似list的操作
    但跟list不同的是，创建时(init)接收的形参是解包后的元素或者OrderedDict，显得更像个初等函数"""
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 支持解包list [OrderedDict]参数，即直接丢OrderedDict进去
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):       # 也支持解包list[module1, module2..]，即直接丢解包后的module进去
                self.add_module(str(idx), module)
    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))
    def __getitem__(self, idx):
        continue
    def __setitem__(self, idx, module):
        """nn.Sequential()依靠__setitem__方法把子module加到父module的属性中去"""
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)
    def __delitem__(self, idx):
        continue
    def __len__(self):
        return len(self._modules)
    def __dir__(self):
        continue
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input
    
class ModuleList(Module): 
    """"借用module类的基本方法模拟出list的效果 (实现extend/append/insert)"""
    def __init__(self, modules):
        if modules is not None:
            self += modules    # 使用自定义iadd重载运算符
    def _get_abs_string_index(self, idx):
        continue
    def __getitem__(self, idx):
        continue
    def __setitem__(self,idx,module):
        continue
    def __delitem__(self, idx):
        continue
    def __len__(self):
        return len(self._modules)
    def __iter__(self):
        return iter(self._modules.values())
    def __iadd__(self):
        return self.extend(modules)
    def __dir__(self):
        continue
    def insert(self):
        continue
    def append(self,module):
        self.add_module(str(len(self)),module)
    def extend(self, modules):
        continue
    
class ModuleDict(Module):
    """"借用module类的基本方法模拟出dict的效果 (实现items/values/keys/update)"""
    def __init__(self, modules):
        if modules is not None:
            self.update(modules)
    def __getitem__(self, key):
        return self._modules[key]
    def __setitem__(self,idx,module):
        self.add_module(key,module)
    def __delitem__(self, idx):
        del self._modules[key]
    def __len__(self):
        return len(self._modules)
    def __iter__(self):
        return iter(self._modules)
    def clear(self):
        continue
    def pop(self,key):
        continue
    def keys(self, modules):
        return self._modules.keys()
    def items(self):
        return self._modules.items()
    def values(self):
        return self._modules.values()
    def update(self):
        continue
    

    
# %%
'''------------------------------module-----------------------------------
Q. 如何创建module容器, 以及组合module容器？
1. 可以用nn.Sequential(*lst), nn.Sequential(), 前者传入的是list解包后的元素，后者传入的是list解包后的OrderedDict()
    后者可以方便增加名称       - 有实现forward()函数
2. 可以用nn.Modulelist(list) - 但forward()需要自己分层写
3. 可以用nn.ModuleDict(dict) - 但forward()需要自己分层写
(3个容器区别：Sequential是一个完整的带forward的module子类，可直接作为children module。而其他2中ModuleList/ModuleDict适合
先创建类，实现forward方法，然后在加入到一个主module中作为children module，好处是继承自module可以默认使用module的方法注册module/参数)

4. 组合module容器
    >可以借用module的方法model.add_module(): 组合后的module作为子模型被加入_modules的字典中作为child_module
'''
import torch.nn as nn
from collections import OrderedDict
# 方式1: 直接输入每一层进sequential
model1 = nn.Sequential(nn.Conv2d(2,2,3),
                       nn.ReLU())
# 方式2: 先list，再解包
layers = [nn.Conv2d(1,2,3),nn.ReLU()]
model2 = nn.Sequential(*layers)
print(model2)
# 方式3: OrderedDict
model3 = nn.Sequential(OrderedDict([('conv1',nn.Conv2d(1,2,3)),
                                   ('re1', nn.ReLU())]))
print(model3)
# 方式4: OrderedDict的简写
model4 = nn.Sequential(OrderedDict(conv1=nn.Conv2d(1,2,3),
                                  re1=nn.ReLU()))
print(model4)
# 方式5: ModuleList, 配合list的所有方法(append/extend/insert)
layers = [nn.Conv2d(2,2,3) for i in range(10)]
model5 = nn.ModuleList(lst)
print(model5)
# 方式6: ModuleDict，配合dict的所有方法(pop/keys()/values()/update)
layers = dict(conv1 = nn.Conv2d(2,4,3),
              conv2 = nn.Conv2d(4,4,3),
              conv3 = nn.Conv2d(4,2,3))
model6 = nn.ModuleDict(layers)
print(model6)

# 基于ModuleList/ModuleDict需要额外实现forward
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ms = nn.ModuleList([nn.Conv2d(2,2,3) for i in range(5)])
    def forward(x):
        for i, m in enumerate(ms):
            if i // 2==0:
                x = m(x)
            else:
                x = nn.ReLU(m(x))
        return x

# 添加子模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.level1 = nn.Sequential(OrderedDict(   # 添加子模型的方式1：直接设置属性
                conv1 = nn.Conv2d(3,64,3),
                bn1 = nn.BatchNorm2d(64),
                relu1 = nn.ReLU()))
        
        self.add_module('level2', nn.Sequential(OrderedDict(   # 添加子模型的方式2：add_module()函数，等效于添加属性
                conv1 = nn.Conv2d(64,64,3),
                bn1 = nn.BatchNorm2d(64),
                relu1 = nn.ReLU())))
        
    def forward(self,x):
        x = self.level1(x)
        x = self.level2(x)
        return x
model = Net()

# 以下为了验证2种添加子模型的方式是完全等价，可以查看_modules/named_modules()/...
print(model)
print(model._modules.keys())
names = []
for name, module in model.named_modules():
    names.append(name)
print('total name len:{}'.format(len(names)))  # 输出主模型/子模型/层模型，只有主模型没有名字
print(names)



'''-----------------------------------------------------------------------
Q.如何便捷获取module的属性？
1. 通过module的3大字典属性_modules, _parameters, _buffers
2. 通过生成器方法modules(), parameters(), buffers(), children() - 只提供值
3. 通过生成器方法named_modules(), named_parameters(), name_buffers(), named_children() - 提供(名称,数值)
方法3使用最方便(因为里边包含了name/param比较全)，而named_children()比named_modules()更方便，因为named_modules()里边模型太完整不方便调用
'''
# 单层模型
import torch.nn as nn
l1 = nn.Linear(2, 2)
l1._modules.keys()
l1._parameters.keys()
l1._buffers.keys()
list(l1.named_modules())
list(l1.named_children())  # 单层没有named_children()

# 多层模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.level1 = nn.Sequential(OrderedDict(
                conv1 = nn.Conv2d(3,64,3),
                bn1 = nn.BatchNorm2d(64),
                relu1 = nn.ReLU()))
        self.level2 = nn.Conv2d(64,3,3)
    def forward(self,x):
        x = self.level1(x)
        x = self.level2(x)
        return x
model = Net()

model._modules.keys()                          # _modules里边保存的是子模型(类似named_children,不过一个是dict一个是iterator)

for module in model.named_modules():           # named_modules()输出所有主模型/子模型/层,其中主模型没名字，其他都预设用属性的名字
    print(module)
for name, module in model.named_modules():     # named_modules()每一个输出元素是tuple(name, module)
    print(name)    

for child in model.named_children():            # named_children()只输出子模型 
    print(child)
for name, module in model.named_children():     # named_children()每一个输出元素是tuple(name, module)
    print(name)
    
for name, param in model.named_parameters():    # named_parameters()每一个输出元素是tuple(name, parameter)
    print(name)


# %%
"""
Q.在pytorch中几个基础模型的创建方式
1. vgg
2. resnet
"""
# ---------vgg-----------------
import torch.nn as nn
class VGG(nn.Module):
    """vgg模型的实现："""
    arch_settings = {
            11: (1, 1, 2, 2, 2),
            13: (2, 2, 2, 2, 2),
            16: (2, 2, 3, 3, 3),
            19: (2, 2, 4, 4, 4)}  # 记住2,2,4,4,4,这个vgg19是最常用
    def make_vgg_layer(self, num_blocks):     # 记住每个block结构(conv3x3+bn+relu)*n + maxpool
        layer = []
        for i in range(num_blocks):
            layer.append(nn.Conv2d(inplane, outplane, 3, padding=padding, dilation=dilation))
            if bn:
                layer.append(nn.BatchNorm2d())
            layer.append(nn.ReLU())
        layer.append(nn.MaxPool2d())
        return layer
    
    def __init__(self,depth, 
                 with_bn=True,
                 dilations=[1,1,1,1,1],
                 out_indics=[],
                 with_last_pool=false):
        self.out_indics = out_indics
        blocks = arch_settings.get(depth)
        layers = []
        for block in blocks:
            layers.extend(make_vgg_layer(block))
        if not with_last_pool:
            layers.pop(-1)
        vgg_layer = nn.Sequential(*layers)
        
        
    def forward(self,x):
        out = []
        features = model(x)
        for layer in self.out_indics:
            out.append(features[layer])
        
# ----------resnet----------------
class BasicBlock:
    """可以先用moduleList实现子模块,，但需要实现forward，然后作为children module加入主模型"""
    def __init__(self):
        pass
    def forward(self):
        pass

class BottleNeck():
    def __init__(self):
        pass
    def forward(self):
        pass
    
class Resnet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))}
    def make_resnet_block(self):
        pass
    def __init__(self):
        pass
    def __forward__(self):
        pass



'''-------------------------------module----------------------------------
Q.在pytorch中conv2d的基本计算过程以及各个参数的作用？
'''
m = nn.Conv2d(3,3,3)
from numpy import random
random.seed(11)
a = random.uniform(0,1, size=(3,5,5))
input = torch.tensor(a)
output = m(input)
print(output)



'''
Q. 对于下采样时conv2d/maxpool的设置区别？
'''
# 作为下采样功能：
# 可以用conv2d/maxpool，对应s=2, 但两者由于计算w/h方式不同所以kernel size一般不同
# conv2d的k-size取3, (w-3+2)/2+1为整数，maxpool的k-size取2, (w-2)/2+1为整数
conv1 = nn.Conv2d()
maxpool1 = nn.MaxPool2d()


    
'''-------------------------------module----------------------------------
Q.在pytorch中mmodule参数初始化方法有哪些，有什么区别？
1. 包含：
2. 
'''
init_weight = ???
net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weight)


# %%
"""Q.在pytorch中的data paralle模块如何实施, 跟常规模型有什么区别？
主要分如下几步：
1. 对model封装，得到Data Parallelmodel, 重写forwrad()，在forward()中定义如下步骤
    >先对数据进行scatter
    >再对模型进行replicas
    >再进行并行输出计算parallel_apply
    >最后再组合输出gather
2. Data parallel模型的不同之处
    >
    >
    > 调试不能直接调试，需要在device数量为1时才能通过forward进入backbone模型本身，否则会因为
      scatter/parallel_apply/gather函数把模型参数分解而无法进入模型执行体本身。
      而由于data parallel模型的forward函数留了一个后门，所以基于device数量为1可以进入模型进行调试
"""
# 熟悉Data Parallel类
# 原始pytorch执行流程：data parallel -> forward -> scatter -> scatter_kwargs -> scatter -> scatter_mape -> Scatter.apply()
# 新的mmdetection流程：MMdataParallel -> forward -> scatter(*) -> scatter_kwargs(*) - > scatter(*) -> scatter_map(*) -> Scatter.forward(*)
class DataParallel(nn.Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        continue
    def forward(self, *inputs, **kwargs):
        """data parallel的计算过程通过forward体现如下：scatter()/replicas()
        /parallel_apply()/gatter()，通过？？进入module"""
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)  # 调用self.scatter
        
        # 这句话挺好的，这样可以在data parallel模式下直接进入模型，否则进入threading模式找不到进入模型的人入口
        # 用来调试单步运行是很好的入口
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        
        # 复制模型：replicate()函数，包含复制主模型(__dict__/_parameters/_buffers/_modules)，复制子模型(_modules/_parameters/_buffers)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])  
        # 并行计算: 首先需要线程锁threading.Lock()，同时创建多线程，
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)
    def scatter(self,inputs, kwargs, device_ids):
        """这是pytorch原有的scatter()"""
        return scatter_kwargs(inputs, kwargs, device_ids, dim =self.dim)
    
    def scatter():
        """这个是子类MMDataParallel的scatter()重写函数，调用的也是重写的scatter_kwargs
        如果是MMDataParallel类(继承DataParallel)则用这个scatter覆盖了父类scatter()
        """
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

def scatter_kwargs():
    """这个也是重写的，但跟pytorch的原函数一样"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

def scatter(inputs, target_gpus, dim=0):
    """这个也是重写的，增加对DataContainer的支持"""
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return OrigScatter.apply(target_gpus, None, dim, obj)  # OrigScatter是pytorch原有Scatter类
        if isinstance(obj, DataContainer):    # 新增对DataContainer的数据对象的支持
            if obj.cpu_only:
                return obj.data
            else:
                return Scatter.forward(target_gpus, obj.data)  # 重写Scatter()类，并加了一个forward()方法
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]
    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None
        
class Scatter(Function): # 原版scatter继承自Function基类
    """这是pytorch的原版Scatter类, 没有apply方法，Scatter.apply()是继承自
    Function父类的父类_C._FunctionBase，也就是C语言写的
    """
    @staticmethod
    def forward():
        continue
    @staticmethod
    def backward():
        continue
    
class Scatter(object):  # 重写Scatter跟原版Scatter没有关系，不是继承
    """这是重写的Scatter类，仅包含一个forward()函数，处理DataContainer数据类型"""
    @staticmethod
    def forward(target_gpus, input):
        input_device = get_input_device(input)
        streams = None
        if input_device == -1:
            # Perform CPU to GPU copies in a background stream
            streams = [_get_stream(device) for device in target_gpus]

        outputs = scatter(input, target_gpus, streams)
        # Synchronize with the copy stream
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs)

# step1: 对model进行wrap成DataParallel(model)
model = DataParallel(model)
# step2: 对input进行scatter(), 最终链接到_C._FunctionBase的C代码，暂时看不到
inputs = Scatter.forward()
# step3: 对model进行复制replica()，也就是复制module的_parameters/_buffers/_modules
def replica():
    continue
# step4: 对多model多参数进行多线程并行计算
def parallel_apply():
    lock = threading.Lock()
    results ={}
    def _worker():
        try:
            with torch.cuda.device(device):
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)
            with lock:              # 如果获得线程锁
                results[i] = output  # 则保存输出
        except Exception as e:
            with lock:
                results[i] = e
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=()) for i, (module, input, kwargs, device) in 
                   enumerate(zip(modules, inputs, kargs_tup, devices))]
        for thread in threads:   # 循环打开，加入，线程
            thread.start()
        for thread in threads:
            thread.join()
    return outputs
# step5: 对多输出进行gather
def gather():
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
    try:
        return gather_map(outputs)
    finally:
        gather_map = None

# 尝试实践一个data parallel模型，看看跟常规模型在模型本身/模型参数上有什么区别？


'''-------------------------------module----------------------------------
Q.在pytorch中distributed模块进行分布式计算的基础是什么？
'''
import torch.distributed as dist
import os
# step1: 分布式训练初始化
def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))
        
def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

# step2: 分布式训练信息汇总
def get_dist_info():
    if dist._initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size





