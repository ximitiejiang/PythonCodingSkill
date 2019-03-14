#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:00:21 2019

@author: ubuntu
"""

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


# %%
