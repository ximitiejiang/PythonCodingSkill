#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 20:08:59 2018

@author: ubuntu
"""

'''----------------------------------------------------------------------
Q. 如何利用断言进行初步的参数判断？
核心理解1：为什么要用断言，是为了防止错误的输入导致了无法检查正误的输出，自己会以为输出是正确的。
          所以先用断言把错误的输入拦截下来，避免程序运行。
理解2：assert跟if一样，都是用于条件判断，assert相当于(确保)，为True则继续执行，为False则报错
理解3：常见条件判断都可以用在if/assert语句中
    - isinstance(data, (list, tuple))
    - str in dict.keys()
    - len(data) == 2
'''
a = 'hello'
assert isinstance(a, str) and 'e' in a



'''----------------------------------------------------------------------
Q. 如何对函数的输入参数进行初步判断，确保输入没问题？
'''

def obj_from_dict(info, parrent=None, default_args=None):

    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type')
    if mmcv.is_str(obj_type):
        if parrent is not None:
            obj_type = getattr(parrent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def test_input_argument(params, parrent=None, default_args=None):
    assert isinstance(obj, dict) and 'type' in params
    
params = dict(type='SGD',lr=0.01)
test_input_argument(params,)


'''----------------------------------------------------------------------
Q. 如何写测试代码？
此处还没理解写个测试代码意义在哪，有空去看https://segmentfault.com/a/1190000007248161
'''
def get_formatted_name(first, last):
    full_name = first + ' ' + last
    return full_name.title()

print("Enter 'q' at any time to quit")
while True:
    first = input('\nPlease give me a first name: ')
    if first == 'q':
        break
    last = input('\nPlease give me a last name: ')
    if last =='q':
        break
    formatted_name = get_formatted_name(first, last)
    print('\tFormatted name:' + formatted_name + '.')


'''----------------------------------------------------------------------
Q. ipdb的几个核心调试命令？
'''
n      # 运行下一句next
s      # 进入函数体内
u      # 往上跳一层查看
d      # 往下跳一层查看
l      # 定位到当前运行语句
b 100  # 在本文件第100行设立断电
c      # 一直运行直到下一个断点


'''----------------------------------------------------------------------
Q. 如何用代码捕捉异常？
1.异常是什么：
2.处理机制：try...except...finally
    重要：所有错误类型都继承自BaseException，
    重要：try-except可以跨层调用
    重要：结合logging模块后，可以在不停止程序的同时，输出错误信息用于调试检查。
'''
# 常规异常判断，不带异常类型
try:
    xx  # 初始操作
except:
    xx  # 异常操作
else:
    xx  # 无异常操作
    
# 用except带多种异常类型那个
try:
    xx
except():
    xx
else:
    xx
# try finally
try:
    

# 用    



'''----------------------------------------------------------------------
Q. 如何使用logging进行调试？
logging的特点：
1. logging不会抛出错误，而且可以输出到文件
2. logging的输出等级：DEBUG < INFO < WARNING < ERROR < CRITICAL，越往后信息量越少
   其中DEBUG/INFO级别的日志用于进行开发或部署调试，WARNING/ERROR/CRITICAL级别的
   日志则用来降低机器I/O压力.
   默认的logging输出是Warning，低于该级别不输出
3. logging的4大组件
    >loggers
    >handlers
    >filters
    >formatters

# 对logging的输出等级进行配置：logging.basicConfig()
logging.basicConfig(filename='logging_test.txt',  # 日志输出文件名，指定后就不会输出到控制台
                    filemode=,  # 日志文件打开方式，默认是'a' (apend)
                    format=,    # 日志格式：
                    datefmt=,   # 日期/时间格式，需要在format中包含时间字段%(asctime)s才有效
                    level=logging.INFO,  # 日志输出等级, 默认级别是WARNING，这里挑低到INFO
                    stream=     # 日志输出目标stream (如sys.stdout, sys.stderr, 网络stream),不能与filename同时提供
                    style=,     # format格式字符串风格('%','{','$')
                    handlers=)  # 多个handlers的可迭代对象
参考：http://www.cnblogs.com/yyds/p/6901864.html
'''
import logging
log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger('new logger')                 # 创建logger对象，没有指定名字，则返回RootLogger, 名字叫root
logger.debug('this is a debug log')
logger.info('this is a info log')
logger.warning('this is warning log')
logger.error('this is error log')
logger.critical('this is critical log')

# 一个标准的logging对象初始化
def init_logger(self, log_dir=None, level=logging.INFO):
    """python logger init, from mmcv lib"""
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=LOG_FORMST, level=level)
    logger = logging.getLogger(__name__)
    if log_dir and self.rank == 0:
        filename = '{}.log'.format(get_time_str())
        log_file = osp.join(log_dir, filename)
        self._add_file_handler(logger, log_file, level=level)
    return logger

# 把logger输出到屏幕
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler('log.txt')  # 会创建一个log.txt文件
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')



