#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 22:52:47 2019

@author: suliang
"""

'''
Q. opencv/cv2的基本操作
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = np.zeros((512,512), np.uint8)
cv2.line(img,(0,0),(200,300),255,5)            # 直线：起点/终点
cv2.line(img,(200,300),(511,300),255,5)
plt.imshow(img,'gray')

img = np.zeros((512,512,3),np.uint8)
cv2.rectangle(img,(100,100),(200,300),(55,255,155),5)  # 矩形：左上角点/右下角点
plt.imshow(img,'brg')

img = np.zeros((512,512,3),np.uint8)
cv2.circle(img,(200,200),200,(55,255,155),5)   # 圆形：圆心/半径
plt.imshow(img,'brg')



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
Q. 图片flip的函数？
- 采用np.flip()作为翻转函数，翻转前后size不变
- axis=0表示沿行变化方向，也就是垂直翻转，axis=1表示沿列变化方向，也就是水平翻转
'''
import numpy as np
from numpy import random
data = random.randint(5,size=(2,4))
data2 = np.flip(data, axis=0)  # axis=0表示沿行变换方向翻，也就是垂直翻
data3 = np.flip(data, axis=1)  # axis=1沿水平翻

img = random.randint(0,255,size=(100,300,3)) # (h,w,c)
img1 = np.flip(img, axis=1)
import matplotlib.pyplot as plt
plt.subplot(121)
plt.title('origi img')
plt.imshow(img)
plt.subplot(122)
plt.title('flipped img')
plt.imshow(img1)


'''-----------------------------------------------------------------
Q. 图片pad的函数
- 用np.ceil判断尺寸
'''
np.ceil(30/4)  # ceil代表上取整
np.floor(30/4) # floor代表下取整
30 % 4    #代表取余
30 // 4   #代表取整的商(等效于np.floor)


'''-----------------------------------------------------------------
Q. 如何绘制bbox？
'''
import cv2
cv2.rectangle(img, left_top, right_bottom, box_color, thickness)

cv2.putText(img, label_text, )


'''-----------------------------------------------------------------
Q. bbox变换？
'''
import numpy as np
a = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
np.clip(a, 5,)

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


