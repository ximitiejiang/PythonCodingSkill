#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 20:08:59 2018

@author: ubuntu
"""
'''
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



'''
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


'''
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


'''
Q. ipdb的几个核心调试命令？
'''
n      # 运行下一句next
s      # 进入函数体内
u      # 往上跳一层查看
d      # 往下跳一层查看
l      # 定位到当前运行语句
b 100  # 在本文件第100行设立断电
c      # 一直运行直到下一个断点

