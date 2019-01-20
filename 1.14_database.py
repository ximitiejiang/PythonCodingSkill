#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 13:27:50 2019

@author: ubuntu
"""

'''------------------------------------------------------------------------
Q.数据库类别？
1. 当前数据库结构都是关系型数据库
2. 付费: Oracle, SQL Server(微软), DB2(IBM), Sybase)
3. 免费: MySQL(最广使用), PostgreSQL(更学术，也不错), SQLite(嵌入式适合桌面和移动端，Python内置)
4. 针对sqlite3数据库：
    >核心2个类Connection和Cursor的对象
'''

# 尝试sqlite3数据库
import sqlite3
conn = sqlite3.connect('dbtest.db')

cursor = conn.cursor()
cursor.execute()
cursor.execute()
cursor.rowcount

cursor.close()  # cursor对象需要关闭
conn.commit()
conn.close()    # connection对象需要关闭

# 更实用的访问方式



'''-------------------------------------------------------------------------
Q. 如何使用MySQL这种主流大型数据库？
'''
