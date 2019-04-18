#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:17:58 2019

@author: ubuntu
"""

"""这部分用于主机上启动任务进程
"""

import random, queue    # multiprocessing.Queue跟python这个queue的区别？？？
from multiprocessing.managers import BaseManager

task_queue = queue.Queue()       # 任务queue容器
result_queue = queue.Queue()     # 结果queue容器

class QueueManager(BaseManager):
    pass

QueueManager.register("get_task_queue", callable=lambda: task_queue)    # 用类方法注册两个方法
QueueManager.register("get_result_queue", callable=lambda: result_queue)

manager = QueueManager(address=('', 5000), authkey=b'abc')  # 创建一个队列管理器：端口5000, 验证码'abc'， 作为主机所以ip地址为空''
manager.start()

task = manager.get_task_queue()      # 获得已经通过manager包装的网络queue对象
result = manager.get_result_queue()

# 放几个任务进去:
#for i in range(10):
#    n = random.randint(0, 10000)
#    print('Put task %d...' % n)
#    task.put(n)                     # 往queue中放入数据,让这些数据通过queue.manager包装的queue对象传递到worker分布式进程中去做计算
# 从result队列读取结果:
print('Try get results...')
for i in range(10):
    r = result.get(timeout=20)     # 从queue中获得数据，设置timeout=10是为了等待分布式worker进程进行计算并传入queue
    print('Result: %s' % r)        # 其中timeout=20是指如果20s之内没有获得结果，则报错退出
# 关闭:
manager.shutdown()
print('master exit.')
