#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:46:40 2019

@author: ubuntu
"""


'''-----------------------------------------------------------------------
Q. 线程与进程的区别，如何使用多线程进行计算加速？
参考：https://www.cnblogs.com/whatisfantasy/p/6440585.html

1. 进程是系统分配资源的最小单元，所以一个程序一般就是一个进程，比如一个播放器，一个浏览器就分别是1个进程
   线程是进程进行运算调度的最小单元，所以一个进程可包含多个线程
   进程个数受限于cpu数量，也就是多少核cpu，每个cpu可启动一个进程
   线程个数一般不受限

2. 并发是指两个或多个事件在同一段时间内发生，而并行是指两个或多个事件同一时刻发生。
   对于python多线程来说，由于有GIL的限制，每个线程需要拿到唯一的GIL执行代码，然后释放GIL给下一个线程，所以python的多线程属于并发，也就是多个线程在同一段时间内发生，但无法做到并行
   对于python多进程来说，有独立的GIL，所以多进程之间是可以并行实现的。
   
3. 虽然python多线程无法并行实现，但实际运行时并不是多进程就比多线程快，因为多进程在切换任务消耗时间比较高，CPU
   原则是：在CPU密集型任务下任务切换比较少，多进程快 (所谓CPU密集型是指比如循环处理，计数运算等，此时CPU占有率较高)，因为CPU密集型任务下如果频繁进程切换自然速度慢下来
   而在IO密集型任务下任务切换比较多，多线程更快 (所谓IO密集型是指比如文件处理，网络爬虫，此时硬盘IO读写较多，时间等待较长)
   针对python的情况，大多数任务下多进程比多线程快，只有在网络请求密集任务上，多线程更快，多进程更占CPU资源。

4. 多线程基本操作过程： 创建线程放入list，启动，加入
    threads = []
    for i in range(10):     # 创建
        threads.append(threading.Thread(target=fn, args=(kk)))
    for thread in threads:  # 启动
        thread.start()
    for thread in threads:  # 加入(相当与等待所有线程结束)
        thread.join()       

   多进程基本操作：创建进程放入进程池，启动
    p = multiprocessing.Pool(10)  # 
    for i in range(0, 10):
        p.apply_async(func=fn, args=(kk))   # 进程池创建进程
    p.close()                               # 进程池关闭
    p.join()                                # 进程池加入(相当与等待所有进程结束)
    

5. 多线程对象的基本方法
   参考：https://www.cnblogs.com/whatisfantasy/p/6440585.html
    thread.start()：启动
    thread.join()：逐个执行每个线程，直到线程执行完(相当与等待结束)
    thread.setName()：设置线程名字
    thread.getName()：获得线程名字
    thread.run()：自定义线程对象的run方法，这是线程被cpu调用自动执行的方法，用于自定义
    thread.setDaemon(True)：设置为守护线程
    
    多进程池对象的基本方法(多进程可以独立操作，也可以用进程池操作，用进程池操作消耗内存空间更少)
    pool.apply_async()：异步(并行)
    pool.apply()：同步(串行)
    pool.close()：等待所有进程结束后，才关闭进程池
    pool.join()：主进程等待所有子进程执行完毕
    
    多进程其他方法
    multiprocessin.cpu_count()：获得本机cpu个数(也就是多核个数)，这也决定了能够开启多少个进程
    
    
3. 进程比线程更好，因为进程更稳定
   多进程可以分布式计算，分布到多台机器上去，但多线程只能分布到一台机器的多个CPU上
'''


# %% 
"""两种任务下，多线程和多进程效率对比
参考：https://www.cnblogs.com/zhangyubao/p/7003535.html
cpu密集型任务下，

"""
import time
import threading
import multiprocessing

max_process = 4
max_thread = max_process

def fun(n, n2):
    """一个CPU密集型计算函数"""
    for i in range(0, n):
        for j in range(0, int(n*n*n*n2)):
            t = i*j

def thread_main(n2):   # 多线程执行，n2表示计算相乘的最大值
    threads = []
    for i in range(0, max_thread):
        t = threading.Thread(target=fun, args=(50, n2))
        threads.append(t)
    start = time.time()
    for i in threads:
        i.start()
    for i in threads:
        i.join()
    print("threads use time: ", time.time() - start, 's')

def process_main(n2):
    p = multiprocessing.Pool(max_process)
    for i in range(0, max_process):
        p.apply_async(func = fun, args=(50, n2))
    start = time.time()
    p.close()
    p.join()
    print("processes use time: ", time.time() - start, 's')

#if __name__ == "__main__":
#    run = True
#    if run:
print('when n=50, n2=0.1:')
thread_main(0.1)
process_main(0.1)

print('when n=50, n2=1:')
thread_main(1)
process_main(1)

print('when n=50, n2=10:')
thread_main(10)
process_main(10)
        
    

# %%
'''Q. 线程锁怎么用？在python中的多线程是否
基本流程： 2步走，第一步主程序创建线程锁/各个线程/打开加入线程，第二步单线程处理函数侦测/执行代码
    1.1 创建一个线程锁，用来传递给任何一个线程，获得线程锁的线程就可以运行，否则就等待：lock = threading.Lock()
    1.2 创建n个线程，每个线程都有自己的worker()线程函数
    1.3 启动每个线程：此时线程开始运行，线程开始搜索，先尝试获得lock(lock.acquire()或者with lock: 这两种方式来判断是否获得lock了)
        如果获得了lock，则开始执行该进程的代码，执行完了之后需要释放lock(lock.release()或者如果是用with lock的方式则会自动释放而无需手动释放，类似于with open(file)这种方式)
    1.4 加入每个线程：此时join指令相当与等待+终止(也就是等到线程执行完，然后终止线程)
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
"""多进程的基础理解
1. 如果一个任务能够用ys=map(func, xs)来解决，他就能够用并行方案解决, 且多进程方案采用的方法其实就类似map()函数
2. 多进程工具multiprocessing.Pool就是python自带的一个多进程创建的简单工具
   + Pool代表进程池，包含3个基本方法map(), imap()和imap_unordered()
   + pool.map(fn, data)，其中fn是进程函数，data是list，用于循环送入fn函数，map()返回的是list
     pool.imap(fn, data)，其中imap返回迭代器
     pool.imap_unordered(fn, data)，其中imap_unordered返回迭代器，且返回的是无序的
   + 注意：Pool使用完毕以后必须关闭，否则进程不会退出，所以最好自动关闭，即采用with...的写法
   + 
"""
import multiprocessing
def f(x):
    return x*x

cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
xs = range(5)
print(pool.map(f,xs))

for ys in pool.imap(f, xs):
    print(ys)

for ys in pool.imap_unordered(f, xs):
    print(ys)


# %%
'''Q. 多进程实现方法
通常是用multiprocess实现多进程，然后基于pytorch的distributed实现多进程分布到一机多卡或多机多卡

1. python多进程与多线程对比
    > 多线程： 线程锁  - 创建多线程 - 启动多线程 - 加入多线程 - 线程子程序抢线程锁
    > 多进程：(无需锁) - 创建多进程 - 启动多进程 - 加入多进程 - (进程子程序自动在每个GPU执行)

2. torch.distributed.launch这是一个启动模块，采用python -m的方式启动：
   $ python -m torch.distributed.launch --nproc_per_node=2 train.py --arg1 --arg2 
   也就相当于通过launch.py这个模块来调用运行我们的train.py文件。
   也就相当于launch.py是一个装饰器，他会预先初始化一些变量，并传递个itrain.py里边的相关代码，比如'RANK'

3. 

'''
# -----------一个简单的创建子进程的过程：用于熟悉父进程和子进程的pid-----------
import os
print('Process (%s) start...' % os.getpid())   # os.getpid()是获得运行该指令的当前进程号，可能是父进程也可能是子进程
pid = os.fork()     # os.fork()是创建一个子进程，并分别在父进程和子进程中返回pid，在父进程中是返回子进程的pid，而在子进程是永远返回0
if pid == 0:  
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))  # 如果pid=0说明是在子进程中。
                                                                                         # 用os.getpid()获得当前进程号，用os.getppid()获得父进程号
else:
    print('I (%s) just created a child process (%s).' % (os.getpid(), pid))   # 如果pid不等于0说明在父进程中，用os.getpid获得当前进程号

# -----------使用python的multiprocessin模块: mutiprocessing是把python的fork做了封装得到的库-----------
from multiprocessing import Process   # Process类用来定义每一个进程对象
import os

def child_proc(name):  # 子进程的运行代码
    print("Running child process %s (%s).."%(name, os.getpid()))

print("Parent process %s"%(os.getpid()))
p= Process(target=child_proc, args=('Proc_Test',))
print("Child process will start.")
p.start()
p.join()
print("child process finished.")


# -----------使用mutiprocessing模块中的Pool类来管理多个子进程-----------
from multiprocessing import Pool
import os, time, random

def sub_process(id):
    print("start running sub_process %s (%s)"%(id, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print("sub_process %s runs for %.2f seconds."%(id, (end-start)))

print("Parent process %s"%os.getpid())   # 初始父进程
ps = Pool()                             # 创建进程池，pool的默认大小是CPU的核数，也就是CPU是几核的就可以有多少个并行进程
                                        # 我的机器CPU是12核心的，所以可以初次同时容纳12个进程
for i in range(18):                       
    ps.apply_async(sub_process, args=(i,))  # 在pool中创建的进程马上就会开始运行，这个比用Process创建的进程更简单，因为Process需要手动start()
                                            # 注意：由于Pool创建时只能一次性装下4个进程，所以第5个进程需要等到前4个进程之一运行完成后，才能加入进程池开始运行
                                            # 这里ps.apply_async() 等效于ps.map(sub_process, args=())也就是相当于map函数的效果
print("waiting child processes run...")
ps.close()    # pool对象包含所有进程，通过close()统一先关闭pool入口，从而不能再添加新的进程process了
ps.join()     # join()方法是在等到所有进程运行完成以后才
print("all the processes done")

# -----------使用mutiprocessing模块中的Queue进行子进程之间的通信-----------
from multiprocessing import Process, Queue
import os, time, random

def write(q):  # 用于把数据写入queue中
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

def read(q):  # 用于把数据从queue中读取出来
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

Q = Queue()                                  # 创建了一个唯一的Q容器，通过他存放数据和释放数据
pw = Process(target=write, args=(Q,))
pr = Process(target=read, args=(Q,))    
pw.start()      # 写进程启动，
pr.start()      # 读进程也启动，此时写进程的for循环还刚刚放入一个A, 由于进程并行处理，只要放入一个值，马上就会读出一个值

pw.join()        # 等待写进程完成然后终止写进程
pr.terminate()   # 这里pr不能用pr.join()来等待结束，因为pr是用while True永远也不会结束，即使已经读取所有queue里边的数据。
                 # 所以要用terminate()强行终止
                 
                 
# -----------使用mutiprocessing模块中的Queue+managers进行多机多进程的分布式通信-----------              
"""
进程: 多个GPU中可以并行跑多个进程
Queue: 多个进程之间通信的桥梁
manager: 用来包装queue，可以在分布式多机上通信

实例参考A15_dist_master.py 和 A15_dist_worker.py 进行分布式数据分发和接收
"""



