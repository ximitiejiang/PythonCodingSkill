#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:46:40 2019

@author: ubuntu
"""


'''
Q. 最简单的实现线程的方式：协程，如何实现协程（也叫用户级线程或绿色线程）？
参考：python cookbook
'''
def countdown(n):
    while n > 0:
        print


'''-----------------------------------------------------------------------
Q. 线程与进程的区别，如何使用多线程进行计算加速？
1. 进程由若干个线程组成，一个进程包含至少一个线程
2. 任何进程默认就会启动一个线程，我们叫主线程，其name属性默认是MainThread，主线程又可以启动新的线程
'''
import time, threading

# 新线程执行的代码:
def loop():
    print('thread %s is running...' % threading.current_thread().name)
    n = 0
    while n < 5:
        n = n + 1
        print('thread %s >>> %s' % (threading.current_thread().name, n))
        time.sleep(1)
    print('thread %s ended.' % threading.current_thread().name)

print('thread %s is running...' % threading.current_thread().name)
t = threading.Thread(target=loop, name='LoopThread')  # 创建新线程
t.start()                                             # 新线程启动
t.join()                                              # 新线程
print('thread %s ended.' % threading.current_thread().name)


'''------------------------------------------------------------------------
Q. 线程锁怎么用？在python中的多线程是否
1. 线程锁
2. GIL(globle interpreter lock全局锁)
3. python多线程的本质：由于有GIL全局锁的存在，每执行100个字节码就会自动释放GIL让别的线程有机会执行
   所以python中的多线程也只能交替执行，只有获得GIL的那个线程在跑。所以pytorch的data parallel也只是
   利用了多个GPU的内存先存储了数据，但每次依然只有1个GPU在运行。
4. 为了提高多GPU的运行效率做到多GPU并行工作：不知道pytorch的分布式训练采用哪种方案？
    >可以多进程操作：每个进程有一个独立GIL，相互不影响
    >可以利用C语言扩展来实现绕开GIL
参考：https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/00143192823818768cd506abbc94eb5916192364506fa5d000
'''
import threading
def change_it(n):
    # 先存后取，结果应该为0:
    global balance
    balance = balance + n
    balance = balance - n

def run_thread(n):
    for i in range(1000000):
        change_it(n)
# 假定这是你的银行存款:
balance = 0
# 启动2个线程分别修改存款数额10000次，理论上无论怎么改都是0
t1 = threading.Thread(target=run_thread, args=(5,))  # 线程1
t2 = threading.Thread(target=run_thread, args=(8,))  # 线程2
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)   # 此时的balance大部分时候为0,但有可能不为0,因为t1在跑没跑完时t2加入，倒是内部的中间变量复用冲突。

# 解决方案：修改run_thead()函数，增加lock
import threading
lock = threading.Lock()
def change_it(n):
    global balance
    balance = balance + n
    balance = balance - n

def run_thread(n):
    for i in range(1000000):
        lock.acquire()    #lock作为全局对象，谁先获得lock,谁就一直有执行权，直到他自己把lock释放掉
        try:
            change_it(n)
        finally:
            lock.release()
balance = 0
t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)


'''-----------------------------------------------------------------------
Q. pytorch的多线程data parallel是如何实现的？
1. 主程序：主要负责创建threads，同时start/join threads
2. 执行函数：主要负责检测lock，
参考：pytorch的parallel_apply.py
'''
import torch
"""需事先准备好modules, inputs, devices, kwargs_tup，每个module都有对应一套数据"""
def _worker(i, module, input, kwargs, device=None):
    """这是每一个子线程的执行函数"""
    try:
        with torch.cuda.device(device):
            output = module(*input, **kwargs)        
        with lock:
            results[i] = output
    except Exception as e:
            with lock:
                results[i] = e
            
lock = threading.Lock()
results = {}
if len(modules) > 1:  # 如果是多GPU对应的多个modules,则创建同等数量的线程，每个GPU一个线程
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

'''------------------------------------------------------------------------
Q. 多进程实现方法
1. 用multiprocess实现多进程
'''
