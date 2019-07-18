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
"""Q. 如何创建一个窗口？
参考：https://www.jianshu.com/p/c9fb28367015
参考：https://blog.csdn.net/ahilll/article/details/81531587
"""
# 创建并显示窗口
win = tkinter.Tk()  # 创建窗口对象
win.title('show')   # 显示标题 
win.geometry('500x300')      # 设置大小和位置
# 这中间就可以布置一些控件然后通过mainloop一起显示

win.mainloop()      # 进入消息循环，此时就开始显示窗口


# %%
"""Q. 如何创建一个label来显示文本信息？
tkinter.Label()
    - win: 已创建的窗口
    - text：要显示的文本
    - bg：背景颜色
    - fg:字体颜色
    - font:字体
    - wraplength:text文本多宽换行
    - justify: 换行后的对齐方式，left/right/..
    - anchor: 位置 n北，e东，w西，s南，c中间，ne东北
"""

# 创建标签文字
win = tkinter.Tk()
win.title("botton")
win.geometry("300x200")
label = tkinter.Label(win, text="this is a word",
                      bg="pink", 
                      fg="red",
                      font=("Arial", 10),
                      width=20,
                      height=10,
                      wraplength=100,
                      justify="left",
                      anchor="ne")
label.pack() # 放置标签
win.mainloop()





# %%
"""Q. 如何创建一个按钮？
button = tkinter.Button(win, text, command, width, height)
    - win：窗口
    - text:显示的文字
    - command:某个函数
    - width:宽度
    - height:高度

"""
def func():
    print("hello world!")

win = tkinter.Tk()
win.title("botton")
win.geometry("300x200")
button = tkinter.Button(win, text="Yes", command=func, width=20, height=10)
button.pack()
win.mainloop()


# %%
"""Q.如何对控件的位置进行精确定位？
有两种方式
1. 绝对定位：用place输入的是控件左上角的绝对坐标
    button.place(x, y, anchor)
        - x
        - y
        - anchor是锚点，也就是对齐点，
2. 相对定位：用
    button.pack(fill=tkinter.Y, side = tkinter.LEFT)
"""
win = tkinter.Tk()
win.title("botton")
win.geometry("300x200")
button1 = tkinter.Button(win, text="Yes", width=20, height=10)
button2 = tkinter.Button(win, text="No", width=20, height=10)
button1.place(x=0, y=0)
button2.pack(fill=tkinter.Y, side=tkinter.RIGHT)

win.mainloop()





# %%
"""Q. 如何创建一个用于用户输入框？
entry = tkinter.Entry(win, show='*')
    - win
    - show: 用户输入后显示的内容，用*可用来输入密码

"""
def show_entry():
    print(entry.get())   # 这里直接从函数体外部获得了entry对象？没有作用域的问题吗？

win = tkinter.Tk()
win.title("entry and button")
win.geometry("300x200")

entry = tkinter.Entry(win)  # 创建输入框
entry.pack()

button = tkinter.Button(win, text="Yes", command=show_entry, width=20, height=10)
button.pack()

win.mainloop()


# %%
"""Q.如何创建文本控件，以及带滚动条文本控件？
1.普通文本控件

2. 带滚动条文本控件

"""





# %%
"""Q.如何创建单选框和多选框控件？
1. 创建单选框控件：

check1 = tkinter.Checkbutton(win, text="money", variable=hobby1, command=updata)

2. 创建多选框控件：
"""




# %%
"""Q.如何创建一个frame框架容器？
1. frame框架用来做什么？
frm = tkinter.Frame(win)

