#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 08:27:33 2019

@author: ubuntu
"""

# %%
"""如何使用urllib进行网络编程？
1. urllib库，用于操作URL的功能库
    + 模块1: urllib.request: 用于提出请求，抓取URL内容
        + request.urlopen(url) 最简单的发起一个请求，发出的请求是一个简单url
          用于打开一个url，返回一个HTTPResponse对象，可进行类似文件对象的操作
          相应的文件对象操作包括f.read(), f.readline(), f.readlines()跟操作文件完全一样
                with request.urlopen('https://..') as f:
                    data = f.read()
                    print(data.decode('utf-8'))
        + request.Request(url)可以进行更复杂操作的请求方式，发出的请求是一个Request对象
          可往该对象添加HTTP头，从而把自己的请求伪装成浏览器请求
                req = request.Request('http://...')
                req.add_header('User-Agent', 'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
                with request.urlopen(req) as f:
                    data = f.read()
                    print(data.decode('utf-8'))
    + 模块2: urllib.parse
    
    + 模块3: urllib.error, 用于产生异常，主要包括2种异常，一种URLError, 另一种HTTPError
    处理异常的好处就是程序不会报错而停止，而会产生提示并且可以自定义下一步操作(比如超时就跳过)
        + 通过try... except来捕捉异常， error.URLError, error.HTTPError
                try:
                    with request.urlopen(url) as f:
                        ..
                except error.URLError as e:
                    print(e.reason)
"""

from urllib import request, parse

# 最简单请求
url = 'http://www.baidu.com'
with request.urlopen(url) as f:
    data = f.read()
    print(data.decode('utf-8'))
    
    
# 通过request对象发出请求
req = request.Request('http://www.baidu.com')
req.add_header('User-Agent', 'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
with request.urlopen(req) as f:
    data = f.read()
    print(data.decode('utf-8'))