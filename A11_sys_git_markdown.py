#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:00:21 2019

@author: ubuntu
"""

# %%
"""基本的markdown语法？
参考：https://www.jianshu.com/p/191d1e21f7ed
"""
# 一级标题
## 二级标题
### 三级标题

**加粗**
*斜体*
***斜体加粗***
~~删除线~~

>引用(产生一条竖线缩进)
>>二级引用(产生两条竖线缩进)

---(水平分割线)
***(水平分割线)

![图片描述](图片地址 "图片title")  # 其中图片描述显示在图片下方，图片title在鼠标移到图片上显示，title可加可不加

[超链接名称](超链接地址 "超链接title")  # 其中超链接名称就是显示的描述，title也是可加可不加

+ 列表内容(产生一个列表小圆点，也可用-代替+)
   + 列表内容(前面加3个空格产生二级列表嵌套，也可用-代替+)

表头|表头|表头
---|:--:|---:   # 这一行是分割表头和内容，中间的-号有一个就行，但为了对齐可以增加n个
内容|内容|内容  # 文字对齐通过:控制，两边都加:则中间对齐，左边加:则左对齐，右边加:则右对齐

`单行代码`

```
多行代码
```

- [] 未处理事项
- [X] 已处理事项
# 基本上就这些了


# %%
"""在linux系统里如何使用sh文件和相关命令编程进行自动运行？
"""



# %%
"""在linux系统用指令创建软链接？
"""
$ cd ssd_detector
$ mkdir data
$ ln -s /home/ubuntu/MyDatasets/coco
$ ln -s /home/ubuntu/MyDatasets/voc/VOCdevkit



# %%
"""在linux系统中如何设计断点续传
"""
$ wget -c https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth



# %%
"""如何在ipython调试软件中设置让图片单独显示而不是嵌入在命令行？
参考：https://blog.csdn.net/mozai147/article/details/79850065
"""
#命令行输入：代表嵌入式显示
%matplotlib inline
#命令行输入：代表独立显示
%matplotlib qt5
# spyder中设置：inline代表嵌入显示，qt5代表独立显示
Tools > Preferences > IPython Console > Graphics > Graphics backend


# %%
"""如何向开源仓库提交PR
"""
git remote -v  # 查看当前目录跟什么仓库建立了连接: 一开始只会跟自己fork的仓库建立连接

git remote add upstream https://github.com/qijiezhao/M2Det  # 跟原始仓库建立fetch/push

git remote -v  # 此时就可以看到跟2个仓库分别建立了fetch/push的连接

git checkout -b mydev  # 创建一个分支，叫做mydev，并自动切换到这个分支
# 然后就可以修改代码了
git add .                # 增加全部更改

git commit -m 'update xxx'   # 提交

git push origin mydev:mydev  # 从我本地mydev分支提交到我的远程mydev分支
# 最后在我的远程仓库点击send pull request


../data/coco/annotations/instances_train2017.json
../data/coco/annotations/instances_train2017.json
