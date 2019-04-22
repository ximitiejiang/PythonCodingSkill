#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:50:54 2019

@author: ubuntu
"""

from math import pi
import numpy as np

def area(r):
    return pi*r*r

if __name__ =='__main__':
    result = area(2.5)
    print("area = %.3f"%result)
    
    d1= [1,2,3,4,5]
    def sum_it(data):
        result=0
        for i in data:
            result +=i
        return result
    sum_result = sum_it(d1)
    print('sum_result=%.1f'%sum_result)