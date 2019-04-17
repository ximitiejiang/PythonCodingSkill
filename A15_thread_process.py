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

if __name__ == '__main__':
    loopit = False
    if loopit:
        print('thread %s is running...' % threading.current_thread().name)
        t = threading.Thread(target=loop, name='LoopThread')  # 创建新线程
        t.start()                                             # 新线程启动
        t.join()                                              # 新线程
        print('thread %s ended.' % threading.current_thread().name)


# %%
'''
Q. 线程锁怎么用？在python中的多线程是否
基本流程： 2步走，第一步主程序创建线程锁/各个线程/打开加入线程，第二步单线程处理函数侦测/执行代码
    1.1 创建一个线程锁，用来传递给任何一个线程，获得线程锁的线程就可以运行，否则就等待：lock = threading.Lock()
    1.2 创建n个线程，每个线程都有自己的worker()线程函数
    1.3 打开每个线程：
    1.4 加入每个线程：此时线程开始搜索，先尝试获得lock(lock.acquire()或者with lock: 这两种方式来判断是否获得lock了)
       如果获得了lock，则开始执行该进程的代码，执行完了之后需要释放lock(lock.release()或者如果是用with lock的方式则会自动释放而无需手动释放，类似于with open(file)这种方式)
    2.1 编写单线程处理函数：侦测线程锁，执行代码
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
# 假定这是你的银行存款:
balance = 100

def run_thread_1(n):
    """一个错误多线程代码，由于没有线程锁，原本应该怎么改都是0
    但实际上只要运行次数够多，结果不为0"""
    global balance
    for i in range(1000000):
        balance = balance + n
        balance = balance - n
    

lock = threading.Lock()    # 增加lock
def run_thread_2(n):
    """一个正确多线程：
    """
    global balance
    for i in range(1000000):
        lock.acquire()    # 持续等待，直到获得才往下执行
        try:
            balance = balance + n
            balance = balance - n
        finally:
            lock.release()

def run_thread_3(n):
    """另一种多线程写法：参考pytorch源码paralell_apply()"""
    global balance
    for i in range(1000000):
        with lock:   # 用with的写法，可以省去lock.release()
            balance = balance + n
            balance = balance - n
                     
def practice(fn, n):
    t1 = threading.Thread(target=fn, args=(n,))
    t2 = threading.Thread(target=fn, args=(n,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(balance)
    
def practice_complex(fn, n):
    """定义10个线程"""
    threads = []
    for i in range(10):
        threads.append(threading.Thread(target=fn, args=(n,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(balance)
    

if __name__ == '__main__':
    thread_test = 4
    if thread_test == 1:   # 第一个多线程代码，结果随时变化
        practice(run_thread_1, 12.34)
    
    if thread_test == 2:      # 第二个多线程代码，结果不变
        practice(run_thread_2, 12.34)
    
    if thread_test == 3:      # 第三个多线程代码，结果不变
        practice(run_thread_3, 12.34)
    
    if thread_test == 4:      # 更多线程代码，结果不变
        practice_complex(run_thread_3, 12.34)     

'''-----------------------------------------------------------------------
Q. pytorch的多线程data parallel是如何实现的？
1. 主程序：主要负责创建threads，同时start/join threads
2. 执行函数：主要负责检测lock，
参考：pytorch的parallel_apply.py
'''
import torch
"""需事先准备好modules, inputs, devices, kwargs_tup，每个module都有对应一套数据"""
def parallel_apply(modules, inputs, kwargs_tup, devices):
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

# %%
'''
Q. 多进程实现方法
通常是用multiprocess实现多进程，然后基于pytorch的distributed实现多进程分布到一机多卡或多机多卡

1. torch.distributed.launch这是一个启动模块，采用python -m的方式启动：
   $ python -m torch.distributed.launch --nproc_per_node=2 train.py --arg1 --arg2 
   也就相当于通过launch.py这个模块来调用运行我们的train.py文件。
   也就相当于launch.py是一个装饰器，他会预先初始化一些变量，并传递个itrain.py里边的相关代码，比如'RANK'

2. 

'''














