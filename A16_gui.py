#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:10:02 2019

@author: ubuntu
"""

# %%
"""Q. 如何使用python3自带的tkinter?
在python3中自带了tkinter，不需要用户再额外安装。
tkinter在旧版python中名字叫Tkinter，但新版本改成了小写tkinter
"""
import tkinter


# %%
"""
参考：https://www.jianshu.com/p/c9fb28367015
"""
win = tkinter.Tk()  # 创建窗口
win.title('show')   # 显示标题 
win.geometry()      # 设置大小和位置

win.mainloop()      # 进入消息循环，可以写控件了

label = tkinter.Label(win, text="this is a word",
                      bg="pink", 
                      fg="red",
                      font=("黑体", 20),
                      width=20,
                      height=10,
                      wraplength=100,
                      justify="left",
                      anchor="ne")

label.pack()