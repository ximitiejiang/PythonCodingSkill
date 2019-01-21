#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 09:48:53 2019

@author: ubuntu
基础：
1. 时间复杂度O: 代表的是算法的复杂程度，而不是算法的执行时间。他是去掉了系数与常数项，保留特征项的一种表示方法。
   主要通过循环次数的n来体现：
   
    常量时间，复杂度O(1)，比如有限次赋值(去掉了系数项，k句为kO(1)写成O(1))
    线性时间，复杂读O(n)，比如单层循环
    平方时间，复杂度O(n^2)，比如双层循环嵌套
    立方时间，复杂度O(n^3)，比如三层循环嵌套
    
    线性时间，复杂度O(n)，线性查找（也就相当与一层循环一遍）
    对数时间，复杂度logn，比如折半查找（原来n次循环，每次减半，循环次数变成了logn，2为底数，所以是O(logn)）
    对数线性时间，复杂度nlogn，比如快速排序？？？
    指数时间，复杂度2^n，比如递归求斐波那且数列？？？
    阶乘时间，复杂度n!，比如？？？    
2. 一般来说：O(1) < O(logn) < O(n) < O(nlogn) < O(n^2) < O(n^3) < O(2^n) < O(n!) 

3. 几种常见数据结构：
    > 队列FIFO: 先进先出，新进来的排队尾，要出去的从队头出去（head队头是出口）
        > 注意双指针技巧：比如交换字符串顺序str='his name is: eason-kevin' -> 'kevin-eason :is name his'
    > 栈LIFO：后进先出，新进来的排队尾，要出去的也从队尾出去（top栈顶是队尾即出入口）
    > 链表：
    > 哈希表：hash表是一种数据结构，通过特定的哈希函数把
    > 二分查找
    > 
3. 算法分类：
    > 数组相关
    > 字符串
    > 链表
    > 数
    > 排序和搜索
    > 动态规划：是指把大问题分成子阶段，后一问题基于前一问题的解答输出，按顺序规划解决
    > 设计问题
    > 数学问题和其他

"""
import numpy as np
from numpy import random

# %% 
"""在python实现一个链表：get(index), addAtHead(val), addAtTail(val), addAtIndex(index, val), deletAtInde(index)
"""
class MyLinkedList():
    def __init__(self):
    def get(self, index):
    def addAtHead(self, val):
    def

obj = MyLinkedList()
param_1 = obj.get(2)
obj.addAtHead(15)
obj.addAtTail(7)
obj.addAtIndex(4,25)
obj.deleteAtIndex(18)

# %% 
"""找到数组中所有两数之和为指定值的整数对，时间复杂度为O(n)
1. 理解时间复杂度概念，O(n)说明只能用1个循环
"""
class FindSum():
    def __init__(self, sum):
        self.sum = sum
    def find(self, data):
        outputs = []
        
        return outputs
        
random.seed(11)        
data = random.randint(1,100,size=(1,50))
model = FindSum(97)
outputs = model.find(data)


# %% 
"""找到数组前k个数？
1. 理解sorted, sort区别：都是针对list而不是array,一个在前面不改变源数据，一个在后面改变原数据
2. list/array的转换：np.array(list), array.tolist()， 转换前后维度相同
""" 
def sort_k(data, k):
    """sort a list and return from 0th to kth"""
    if isinstance(data, np.ndarray):
        data = data.tolist()
    return(sorted(data)[:k])

random.seed(11)
data = random.randint(1,50, size =(20,))
outputs = sort_k(data, 6)
print(outputs)

# %%  LC-122
"""输入一个数组，第i个元素就是第i天股票价格，买入一次股票就要抛出一次，不能连续买入或连续卖出。
需要定义一个方法，让自己获利最多
1. 最简洁算法理论：需要理解低就买，高就卖的原则，也就是说只要后n天的价格都高，这个利润是肯定可以拿到的，因为可以延迟卖出
    简洁算法背后的算法理论是贪心算法：即每一步都是当前最好的选择，每一步都很贪心的获得最大利益，所以叫贪心算法
2. 如果暴力法，时间复杂度为O(n^n)，可见简洁算法的好处，从解决问题的本质出发，时间复杂度也降为O(n)
"""
def maxprofit(prices):
    """贪心算法: 每次选择卖出获利最大的前提才买入"""
    profit = 0
    for i, p in enumerate(prices):
        if i == len(prices)-1:
            break
        elif prices[i+1] > p:
            profit += prices[i+1] - p
    return profit

#prices = [7,1,5,3,6,4]
prices = [1,2,3,4,5]
profit = maxprofit(prices)
print(profit)

# 一个类似的贪心算法题
# 一个背包，容量M=150，如果有7个物品可分割成任意大小，要求尽可能让装入背包的物品总价值最大，但不超过总容量
# 物品(A,B,C,D,E,F,G),重量(35,30,60,50,40,10,25),价值(10,40,30,50,35,40,30)
def maxweight(weights, values, M):
    """贪心算法：每次选择总量最小/价值最大的物品装入，
    但要注意贪心算法容易漏掉不符合贪心算法但却符合问题需求的补充需求，比如本题最后的一个数据
    也要注意贪心算法有可能不符合题目的解法要求
    """
    import numpy as np
    compare = []
    for wt, vl in zip(weights, values):
        compare.append(vl/wt)
    indics = np.argsort(compare)[::-1]  # from max to min
    
    M_real = 0
    for idx in indics:
        if M_real + weights[idx] <= M:
            M_real += weights[idx]
        elif idx == len(indics)-1:
            break
    return M_real
    
weights = [35,30,60,50,40,10,25]
values = [10,40,30,50,35,40,30]
print(maxweight(weights, values, 150))    

# %%  LC-136
"""数组中只有一个元素出现过1次，其他都出现过2次，找出只出现过一次的元素？
1. 差集：差异的数据，用a^b
2. 并集：合并不重复的数据，用a.union(b)
3. 交集：重复的数据
"""
def find_one_time(data):
    a = 0
    for b in data:
        a = a^b
    return a
    
data = [2,2,1,3,3,6,6,7,7]
find_one_time(data)


# %%  LC-151
"""把一串单词字符串翻转顺序输出？
1. 字符串的split操作：所有str的函数都后置，很方便，比如s.split(), s.strip()
2. 字符串的组合操作：类似list的 + 重载运算符
"""
def reverswords(s):
    s = 'the sky is blue'
    splited = s.split()[::-1]
    result = ''
    for i in splited:
        result += i + ' '
    return result[:-1]
s = 'the sky is blue'
reverswords(s)


# %%  LC-152
"""找到一个数组中，乘积最大的连续子序列？注意必须是连续的子序列，至少包含1个数
1. 考察双重遍历的思想：第一重遍历，从最左边开始取初始值，第二重遍历，从第一重遍历的起点开始取每一个相乘的值
2. 这个方法似乎2个循环嵌套导致效率不高耗时较多，
"""
def maxproduct(nums):
    max_list = [nums[0]]
    for i, _ in enumerate(nums):
        mul = 1
        for j in nums[i::]:
            mul *= j
            if mul>max_list[-1]:
                max_list.append(mul)
    print(max_list[-1])    
    
#nums = [2,3,-2,4]
#maxproduct(nums)   
nums = [0,2]
maxproduct(nums)


# %%  LC-155
"""设计一个最小元素堆栈，能够支持push, pop, top操作，
其中push(x)为推入栈中，pop()为删除栈顶，top()为获取栈顶，getMin()为获取最小

1. 对list的基本操作
2. 
"""
class MinStack(object):

    def __init__(self):
        self.result = []
        self.min = []        
    def push(self, x):
        self.result.append(x)
        if len(self.min) == 0:
            self.min.append(x)
        else:
            if x < self.min[-1]:
                self.min.append(x)
            else:
                self.min.append(self.min[-1])        
    def pop(self):
        self.min.pop()
        return self.result.pop()        
    def top(self):
        return self.result[-1]       
    def getMin(self):
        return self.min[-1]

obj = MinStack()
obj.result
obj.push(6)
obj.result
obj.pop()
obj.push(-2)
obj.push(7)
obj.top()
obj.result
obj.getMin()


# %%  LC-162
"""找到数组中的所有峰值元素：即其值大于左右相邻元素的值
1. 需要考虑多种长度尺寸list的可能性
2. 这种寻找峰值的算法的应用场景：比如在买卖股票的题目中，需要在峰段卖出，谷段买入
"""
def findPeakElement(nums):
    if len(nums) == 1:
        return [0]
    peak = []
    for i, num in enumerate(nums):
        if i == 0 and len(nums)>1:
            if num > nums[i+1]:
                peak.append(i)
        elif i == len(nums)-1 and len(nums)>1:
            if num > nums[i-1]:
                peak.append(i)
        else:
            if num > nums[i-1] and num > nums[i+1]:
                peak.append(i)
    if len(peak) == 1:
        return peak[0]
    else:
        from numpy import random
        return(random.choice(peak))
    
nums = [1,2,1,3,5,6,4]
results = findPeakElement(nums)
print(results)


# %%  LC-729
"""实现一个Calendar类来存放日程，要包含一个book预订方法，输入[start,end)前闭后开
如果日程不冲突，返回true,否则返回false, start和end的取值范围是实数[0,10^9]

1. 实数的科学计数法表示方法：1.0e9代表实数乘以10的9次方
2. 
"""
class MyCalendar():
    def __init__(self):
        self.booked = [(0.0,0.0), (1.0e9+1,1.0e9+1)]
    def book(self, start, end):
        for i, bkd in enumerate(self.booked):
            if start >= bkd[1] and end <= self.booked[i+1][0]:
                self.booked.insert(i+1,(start, end))
                return True
            elif i == len(self.booked)-1:
                return False
            
obj = MyCalendar()
print(obj.book(10,20))
print(obj.book(20,35))
print(obj.book(25,40))
        
        