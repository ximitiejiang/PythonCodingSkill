#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 22:52:47 2019

@author: suliang
"""

'''-----------------------------------------------------------------
Q. RGB与HSV的区别？
'''



'''-----------------------------------------------------------------
Q. 图片处理中几个变换基础以及读取和显示的方法差别？
1. 图片几个核心概念
    **像素**
        >一张图每个位置点用一个数字，整个图片尺寸就是像素个数(w,h)
    **色彩表示方式**
        >RGB(常用)
        >HSV(常用)：hue(色相), saturation(饱和度), value(色调)
        >HSB: hue(色相), saturation(饱和度), brightness(明度)
        >HSL: hue(色相), saturation(饱和度), lightness(亮度)
    **颜色顺序**
        >rgb顺序
        >bgr顺序
    **维度顺序**这只针对rgb/bgr这种已经分解分层的RGB图
        >(h,w,c)，大部分的应用
        >(c,h,w)，少部分应用(比如pytorch)
2. 图片读取和显示方案的差别
    
'''



'''-----------------------------------------------------------------
Q. 图片处理过程？
'''
path=''
# 1. read - (h,w,c) - bgr(0~255)

# 2. extra augment - (h,w,c) - bgr(0~255)

# 3. scale or resize - (h,w,c) - bgr(0~255) - 影响bbox

# 4. normalization - (h,w,c) - bgr(-2.x~2.x)

# 5. bgr to rgb - (h,w,c) - rgb(-2.x~2.x)

# 6. padding - (h,w,c) - bgr(-2.x~2.x) - 影响bbox

# 7. flip or rotate - (h,w,c) - bgr(-2.x~2.x) - 影响bbox

# 8. transpose - (c,h,w) - bgr(-2.x~2.x)

# 9. to tensor - (c,h,w) - bgr(-2.x~2.x) - 影响bbox


