#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 22:52:47 2019

@author: suliang
"""
"""
Q. 如何让matplotlib在spyder显示的图片单独窗口显示而不是显示在命令行
"""
# 方式1: spyder中设置
# Tools > Preferences > IPython Console > Graphics > Graphics backend

# 方式2
# %matplotlib qt5



'''-----------------------------------------------------------------------
Q. 读取图片/显示图片/写入图片？
'''
import cv2
import matplotlib.pyplot as plt
# 读取：一般用cv2.imread(), 直接得到bgr图
img = cv2.imread('test/test_data/messi.jpg',1) # 1为彩色图，0为灰度图，-1为？
# 显示：一般用plt.imshow(),也可用cv2自带的
plt.imshow(img[...,[2,1,0]])
# 写入图片
cv2.imwrite('messigray.png',img)

'''-----------------------------------------------------------------
Q. cv2的图片读写
- 读图：cv2.imread(path): 应用广泛，能直接得到bgr，比PIL少需要一步转换
- 写图：cv2.imwrite(path,img,params)
- 显示图： plt.imshow(path):这个比cv2.imshow()更方便，不用延时检验之类的操作
'''
import matplotlib.pyplot as plt
path = 'test/test_data/messi.jpg'
img = cv2.imread(path,1)

cv2.imwrite(file_path, img, params)



'''-----------------------------------------------------------------------
Q. 读取和显示视频
参考：https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html#display-video
'''
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)   # 创建视频帧捕捉对象, 0为device index，会同时打开摄像头

while(True):
    if not cap.isOpened(): # 有时cap没有初始化capture对象，可通过isOpen()进行检查，如果不对则重新初始化
        cap.open()    
    ret, frame = cap.read()     # Capture frame-by-frame, 返回True/False和帧，
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    break
cap.release()  # 释放捕捉，会同时关闭摄像头

# 播放视频
cap = cv2.VideoCapture('vtest.avi')
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    break
cap.release()

# 保存视频
cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()



'''------------------------------------------------------------------------
Q. 如何在opencv控制鼠标
'''
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)


'''------------------------------------------------------------------------
Q. 如何分解/组合/调整各个通道
'''
b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))

#另一种方式
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]


'''------------------------------------------------------------------------
Q. opencv/cv2的基本画图：直线，矩形，圆形？
1. 在opencv中绘制等效于在img上直接绘制并跟img合成一张图，所有命令需要传入img
2. 显示建议用plt.imshow，比用cv2的更简单，不需要延时检测
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = np.zeros((512,512), np.uint8)
cv2.line(img,(0,0),(200,300),(255,0,0),5)            # 直线：起点/终点
cv2.line(img,(200,300),(511,300),(0,0,255),5)
plt.imshow(img,'gray')

img = np.zeros((512,512,3),np.uint8)
cv2.rectangle(img,(100,100),(200,300),(55,255,155),5)  # 矩形：左上角点/右下角点
plt.imshow(img,'brg')

img = np.zeros((512,512,3),np.uint8)
cv2.circle(img,(200,200),200,(55,255,155),5)   # 圆形：圆心/半径
plt.imshow(img,'brg')

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)



'''-----------------------------------------------------------------
Q. RGB与HSV与gray的区别？
1. 主要有3中颜色空间，一种RGB，一种HSV，一种灰度
    RGB: 0-255
    gray: 
    HSV:Hue色调范围[0,179], Saturation饱和度范围[0,255]，Value明度范围[0,255]
2. gray灰度应用范围
3. HSV应用范围：

'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datasets.color_transforms import bgr2gray, bgr2hsv, hsv2bgr, bgr2rgb, rgb2bgr

img1 = cv2.imread('test/test_data/opencv_logo.png',1) # bgr: 0-255
plt.imshow(img1[...,[2,1,0]])

# rgb基本色???
red = np.uint8([[[255,0,0]]])
green = np.uint8([[[0,255,0]]])
blue = np.uint8([[[0,0,255]]])
black = np.uint8([[[0,0,0]]])
white = np.uint8([[[255,255,255]]])

# to gray
img2 = bgr2gray(img1)                                 # gray: 0-255
plt.imshow(img2)
# to hsv: hsv比rgb更容易表示一个颜色
img3 = bgr2hsv(img1)                                  # hsv: 0-255
plt.imshow(img3)

# 在hsv下提取蓝色: 先通过bgr基本色找到对应的hsv数据，
# 然后对h+-10作为主要决定范围即可，s/v两项可以放很宽都行
blue_rgb = blue                         # rgb蓝色 (0, 0, 255)
blue_hsv = bgr2hsv(rgb2bgr(blue_rgb))   # hsv蓝色 (120,255,255)

lower_blue = np.array([110,50,50])    # 所以取120的上下10, s/v的值可以往下取很小到50
upper_blue = np.array([130,255,255])

mask = cv2.inRange(img3, lower_blue, upper_blue)  # inRange函数让低于阈值和高于阈值的都变为0，在之间的变为255
                                                  # 生成的mask是一张0/255的二维数据
res = cv2.bitwise_and(img3,img3, mask= mask)   # 用mask与原图进行相与
                                               # 
plt.subplot(141)
plt.imshow(img1[...,[2,1,0]]) # bgr转成rgb
plt.subplot(142)  
plt.imshow(img3)  # hsv
plt.subplot(143)
plt.imshow(mask)  # mask
plt.subplot(144)
plt.imshow(hsv2bgr(res)[...,[2,1,0]])   # res from hsv to bgr to rgb


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



'''-------------------------------------------------------------------------
Q. 如何定义图片的位置？
1. 图片左上角0，0， 水平向右为w正方向，垂直往下为h正方向
2. 读取进来一般是(h,w,c)或者(w,h)两种尺寸的图片
3. ROI是指region of intrest
'''



'''-------------------------------------------------------------------------
Q. 图片的混合操作？
1. 这里cv2也是重载了运算符add和addWeighted，用来把两张图片按一定比例混合成一张图片
2. 两张图可以不一样大小，但必须相同通道数
'''
img1 = cv2.imread('test/test_data/test1.jpg',1)
img2 = cv2.imread('test/test_data/test2.jpg',1)

img3 = cv2.add(img1, img2)
plt.imshow(img3[...,[2,1,0]])

img4 = cv2.addWeighted(img1,0.7,img2,0.3,0)
plt.imshow(img4[...,[2,1,0]])


'''------------------------------------------------------------------------
Q. 图片部分ROI的抠图以及组合？
1. 阈(yu)值的概念：
2. ret,mask = cv2.threshold(src,thresh,maxval,type)，
其中src为源图，需要时灰度图，thresh是阈值，maxval是最大值，
type是转换模式(cv2.THRESH_BINARY代表)

参考：https://blog.csdn.net/weixin_35732969/article/details/83779660
'''
img1 = cv2.imread('test/test_data/messi.jpg',1)
img2 = cv2.imread('test/test_data/opencv_logo.png',1)

rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)  #转成灰度图作为mask
plt.imshow(img2gray)

ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

plt.imshow(img1)


'''-------------------------------------------------------------------------
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
- 用cv2的cv2.rectangle()是把图片直接先画在img上，然后就可以直接显示img即可
关键理解：图片的坐标系是左上角是(0,0), 往右是x正方向，往下是y正方向(可通过画图看出来)
    而bbox的坐标形式[xmin,ymin,xmax,ymax]对应就是xmin,ymin显示为bbox的左上角，xmax,ymax显示为bbox的右下角
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





'''
Q. 为什么cv2经常报一个错：TypeError: Layout of the output array img is 
incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)
参考：https://www.cnblogs.com/ocean1100/p/9496775.html
https://www.jianshu.com/p/cc3f4baf35bb
核心原因：cv2里边很多函数都是一种输入等于输出的函数，比如传入cv2.rectangle,
传入img,最终也会输出img，而如果输入的是img.transpose()这是一种浅拷贝，但会改变
内存的方式变为不连续，而输出以后是原来的img，他是连续内存，这就造成了冲突。
查看内存方式是img.flags(这是numpy提供的函数)
所以 解决办法是确保输入和输出是一致的: 先做完对img的所有操作(numpy/transpose/astype...)
然后img = img.copy()，用这个d.copy()对array完成深度拷贝得到新的img去做绘图就不会报错！

'''
import cv2
import numpy as np
# --------------------错误实例：-------------------------------
img = cv2.imread('repo/test.jpg')[:,:,::-1]  #bgr2rgb, 浅拷贝
#拷贝img至img_copy
img_copy = img.copy()
#输入要画的框
box = np.array([0, 12, 13, 18, 2, 20, 3, 40])
#画框
cv2.polylines(img_copy[:, :, ::-1], box.astype(np.int32).reshape(-1,1,2),
              isClosed=True, color=(255,255,0), thickness=2)
# --------------------正确实例：-------------------------------
img = cv2.imread('repo/test.jpg')
#拷贝img至img_copy
img = img[:,:,::-1].copy()
#输入要画的框
box = np.array([0, 12, 13, 18, 2, 20, 3, 40])
#画框
cv2.polylines(img, box.astype(np.int32).reshape(-1,1,2),
              isClosed=True, color=(255,255,0), thickness=2)



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


'''-----------------------------------------------------------------
Q. 图片处理过程？
'''