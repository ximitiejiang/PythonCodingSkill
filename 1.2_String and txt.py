#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:26:11 2018

@author: suliang
"""

'''
Q: 字符串的基本操作函数？
1. 字符串操作函数都为后置式，比较好记

'''
# 大小写操作
str = 'Hello'
str.upper()    # 全部大写
str.lower()    # 全部小写
str.swapcase()    # 大小写互换
str.title()    # 首字母大写

# 去除指定字符和空格
str = ' he is good guy '
str.strip()     # 去两边空格
str.strip('\n') # 去掉尾端的回车
str.lstrip()    # 去左边空格
str.rstrip()    # 去右边空格

lst = str.split()     # 默认按空格分隔
str1 = 'a,b,c,d'
str2 = str1.split(',')    # 按指定字符分割字符串为数组

# 判断字符串
str = 'Everyday is a new day!'
str.startswith('E')    # 是否以start开头
str.endswith('!')    # 是否以end结尾
str.isalpha()    # 是否全字母
str.isdigit()    # 是否全数字
str.islower()    # 是否全小写
str.isupper()    # 是否全大写
str.istitle()    # 判断首字母是否为大写
str.isspace()    # 判断字符是否为空格
