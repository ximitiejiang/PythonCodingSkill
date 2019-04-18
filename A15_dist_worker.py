#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:17:58 2019

@author: ubuntu
"""

"""这部分用于从机上启动任务进程（也可以在同一主机上运行, 比如开2个terminal, 一个运行master.py, 另一个运行worker.py）
"""

import time
from queue import Queue    # multiprocessing.Queue跟python这个queue的区别？？？
from multiprocessing.managers import BaseManager

class QueueManager(BaseManager):
    pass

QueueManager.register("get_task_queue")    # 由于假定主机上已经定义了queue并注册了两个名字的queue，所以这里只提供名字就可以了
QueueManager.register("get_result_queue")

server_addr = "127.0.0.1"
print("Connecting to server %s..."%server_addr)

manager = QueueManager(address=(server_addr, 5000), authkey=b'abc')  # 连接到服务器，也就是master.py运行的机器上，端口/验证码需要跟master.py设置的一致
manager.connect()  # 开始连接

task = manager.get_task_queue()      # 获得已经通过manager包装的网络queue对象
result = manager.get_result_queue()

# 提取
for i in range(10):
    try:
        n = task.get(timeout=1)     # 提取分布系统传递过来的值
        print('run task %d * %d...' % (n, n))
        r = '%d * %d = %d' % (n, n, n*n)  # 计算并生成一个字符串
        time.sleep(1)
        result.put(r)                    # 把结果放入网络queue对象
    except Queue.Empty:
        print('task queue is empty.')
# 处理结束:
print('worker exit.')
