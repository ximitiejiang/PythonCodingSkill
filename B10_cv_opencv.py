#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 22:52:47 2019

@author: suliang
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datasets.color_transforms import bgr2gray, bgr2hsv, hsv2bgr, rgb2bgr,bgr2rgb,gray2bgr

# %%
"""Q. 如何让matplotlib在spyder显示的图片单独窗口显示而不是显示在命令行
"""
# 方式1: spyder中设置
# Tools > Preferences > IPython Console > Graphics > Graphics backend

# 方式2
# %matplotlib qt5


# %%
'''Q. 图片处理过程？
1. 图片正向处理过程中：主要就是读取hwc/bgr，然后norm化，rgb化，chw化，tensor化
2. 如果要逆向这个过程，需要改变的包括：
    > 逆tensor化
    > 逆chw化:  这一步要先做，否则可能导致后边广播机制不成功
    > 中间这步rgb化不需要做，因为plt.imshow正好只认rgb(不像cv2.imshow认的是bgr)
    > 逆norm化： img * std - mean, 这一步要注意是数据从float(+-2.x)变int(0-255)
      由于数据集的归一化是到N(0,1)但数值是超过(0,1)这就意味着直接用float显示不对，
      所以最好逆归一化后，先格式转换到int32再截断到(0-255,否则会导致显示不出来
      np.clip((img*std+mean).astype(np.int32), 0, 255)
    > 显示plt.imshow(): 要求hwc/rgb， 要求float(0-1)或int(0-255)，否则无法显示(即使自动clip但依然无法显示)
'''
path=''
# 1. read - (h,w,c) - bgr(0~255)
# 2. extra augment - (h,w,c) - bgr(0~255)
# 3. scale or resize - (h,w,c) - bgr(0~255) - 影响bbox
# 4. normalization - (h,w,c) - bgr(-2.x~2.x)     这一步是归一化的一种，归一化包括了(规则化到标准正态分布，归一化到数值0-1)比如pytorch中to_tensor归一化采用的方式是把数据转到(0-1)之间
# 5. bgr to rgb - (h,w,c) - rgb(-2.x~2.x)
# 6. padding - (h,w,c) - bgr(-2.x~2.x) - 影响bbox
# 7. flip or rotate - (h,w,c) - bgr(-2.x~2.x) - 影响bbox
# 8. transpose - (c,h,w) - bgr(-2.x~2.x)
# 9. to tensor - (c,h,w) - bgr(-2.x~2.x) - 影响bbox


# %%
"""Q. 读取图片/显示图片/写入图片？
1. 最常用显示图片的是plt.imshow(), 要求：rgb, (h,w,c)
"""
# 读取：一般用cv2.imread(), 直接得到bgr图
def imread():
    img = cv2.imread('test/test_data/messi.jpg',1) # 1为彩色图，0为灰度图

    # 显示：一般用plt.imshow(),也可用cv2自带的
    """注意plt.imshow默认是基于rgb的颜色空间显示，
       >如果是bgr则需转成rgb
       >如果是gray则需要指定cmap(colormap), cmap = plt.cm.gray, 或cmap='gray'
    """
    plt.imshow(img[...,[2,1,0]], cmap='gray')
    # 写入图片
    cv2.imwrite('messigray.png',img)


'''-----------------------------------------------------------------
Q. cv2的图片读写
- 读图：cv2.imread(path): 应用广泛，能直接得到bgr，比PIL少需要一步转换
- 写图：cv2.imwrite(path,img,params)
- 显示图： plt.imshow(path):这个比cv2.imshow()更方便，不用延时检验之类的操作
'''
def img_write():
    path = 'test/test_data/messi.jpg'
    img = cv2.imread(path,1)

    cv2.imwrite(path, img, params)



'''-----------------------------------------------------------------------
Q. 读取和显示视频
参考：https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html#display-video
1. cap = cv2.VideoCapture(): 可用来打开摄像头，或者打开一段视频
2. cap.read(): 返回
'''
import matplotlib.pyplot as plt

def show_cam():
    cap = cv2.VideoCapture(0)   # 创建视频帧捕捉对象, 定义0为device index，会打开编号为0的摄像头
    while(True):
        if not cap.isOpened():  # 有时cap没有初始化capture对象，可通过isOpen()进行检查，如果不对则重新初始化
            cap.open()    
        ret, frame = cap.read()     # Capture frame-by-frame, 返回True/False和帧
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray)
        break
    cap.release()  # 释放捕捉，会同时关闭摄像头

# 播放视频
def video_show():
    cap = cv2.VideoCapture('vtest.avi')
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray)
        break
    cap.release()

# 保存视频
def save_video():
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

if __name__ == '__main__':
    show_video = False
    if show_video:
        save_video()
        


'''------------------------------------------------------------------------
Q. 如何在opencv控制鼠标
'''
def show_event():
    events = [i for i in dir(cv2) if 'EVENT' in i]
    print(events)


'''------------------------------------------------------------------------
Q. 如何分解/组合/调整各个通道
'''
def demo1():
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
def demo2():
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
    RGB: 共3个通道(r,g,b)，每个通道数值范围0-255，每一个单通道都是一个灰度通道(0-255)，3个灰度通道经过组合处理才能得到彩色
         对应通道数字越大越靠近本通道[255,0,0]就是红色，[0,255,0]就是绿色，[0,0,255]就是蓝色，[0,0,0]就是黑色，[255,255,255]就是白色
    gray: 只有1个通道，数值范围0-255，
         数字越小，灰度越黑，0就是黑色，1就是白色。
    HSV: 共3个通道，Hue色调/数值范围[0,179], Saturation饱和度数值范围[0,255]，Value明度数值范围[0,255]
         h数字(0是红色，60是绿色，120是蓝色), s数字(0-255), v数字(0-255)
         
2. gray灰度应用范围
3. HSV应用范围：

'''
def demo3():
    img1 = cv2.imread('test/test_data/opencv_logo.png',1) # bgr
    plt.imshow(img1[...,[2,1,0]])

    # rgb基本色：下面代表一个像素点的3层数字，即r层g层b层3个数字
    red = np.uint8([[[255,0,0]]])
    green = np.uint8([[[0,255,0]]])
    blue = np.uint8([[[0,0,255]]])
    black = np.uint8([[[0,0,0]]])
    white = np.uint8([[[255,255,255]]])

    img2 = img1[...,[2,1,0]]  # rgb
    r,g,b = cv2.split(img2)   # (222,180,3) -> 3x (222,180)
    plt.imshow(g)             # 每一路都类似一张2d的gray图，范围0-255

    # to gray
    img3 = bgr2gray(img1)            # gray: 0-255
    plt.imshow(img3, cmap='gray')

    # to hsv: hsv比rgb更容易表示一个颜色
    img4 = bgr2hsv(img1)             # hsv
    plt.imshow(img4)
    h, s, v = cv2.split(img4)

    img5 = cv2.imread('test/test_data/messi.jpg')
    img5 = bgr2hsv(img5)
    h, s, v = cv2.split(img5)        # h数值范围是0-180度(红色数值为0, 绿色为60，蓝色为120)

    # 在hsv下提取蓝色: 先通过bgr基本色找到对应的hsv数据，
    # 然后对h+-10作为主要决定范围即可，s/v两项可以放很宽都行
    blue_rgb = blue                         # rgb蓝色 (0, 0, 255)
    blue_hsv = bgr2hsv(rgb2bgr(blue_rgb))   # hsv蓝色 (120,255,255)

    lower_blue = np.array([110,50,50])    # 所以h取120的上下10, s/v的值可以往下取很小到50
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
        >一张图每个位置点用一个数字，整个图片尺寸就是像素个数(h,w)
    **色彩表示方式**
        >RGB(常用)
        >HSV(常用)：hue(色相), saturation(饱和度), value(色调)
        >HSB: hue(色相), saturation(饱和度), brightness(明度)
        >HSL: hue(色相), saturation(饱和度), lightness(亮度)
    **颜色顺序**
        >rgb顺序
        >bgr顺序
    **维度顺序**这只针对rgb/bgr这种已经分解分层的RGB图
        >(h,w), 灰度图
        >(h,w,c)，大部分的应用,比如opencv
        >(c,h,w)，少部分应用(比如pytorch)
        (注意：h,w的定义方式，跟array都是一致的，即先行数(h)再列数(w))
2. 图片读取和显示方案的差别
    > plt.imread()读取的rgb, cv2.imread()读取的是bgr
    > plt.imshow()按照rbg方式显示，cv2.imshow()需要增加延时和按键监测    

3. 如何定义图片的位置？
    >图片左上角0，0， 水平向右为w正方向，垂直往下为h正方向
    >读取进来一般是(h,w,c)或者(h,w)两种尺寸的图片, 这与所有的array是一致的，即先行数再列数
    >ROI是指region of intrest
'''



'''-------------------------------------------------------------------------
Q. 图片的混合操作？
1. 这里cv2也是重载了运算符add和addWeighted，用来把两张图片按一定比例混合成一张图片
2. 两张图可以不一样大小，但必须相同通道数
'''
def mix_img():
    img1 = cv2.imread('test/test_data/test1.jpg',1)
    img2 = cv2.imread('test/test_data/test2.jpg',1)

    img3 = cv2.add(img1, img2)
    plt.imshow(img3[...,[2,1,0]])

    img4 = cv2.addWeighted(img1,0.7,img2,0.3,0)
    plt.imshow(img4[...,[2,1,0]])


'''------------------------------------------------------------------------
Q. 用opencv如何对图像进行尺寸变换？
'''
"""位置变换"""
def warp_img():
    img = cv2.imread('test/test_data/messi.jpg',0)
    rows, cols = img.shape
    M = np.float32([[1,0,100],[0,1,50]])  # 建立平移矩阵，最后一列代表x,y的平移量
    dst = cv2.warpAffine(img, M, (cols,rows))
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(dst)

"""尺寸缩放：这个命令也很强大，是mmdetection里用来rescale和resize的底层命令"""
def resize_img():
    img = cv2.imread('test/test_data/messi.jpg',0)
    h,w = img.shape[:2]
    res = cv2.resize(img, (2*w, 2*h), interpolation=cv2.INTER_CUBIC)  # 直接用tuple输入实际的w,h
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(res)

"""图片旋转： 这个旋转命令很强大，可以指定旋转中心，旋转角度，缩放比例, 也是mmdetection的rotate底层命令""" 
def rotate_img():
    img = cv2.imread('test/test_data/messi.jpg',0)
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1) # 建立旋转矩阵：输入旋转中心，旋转角度，比例为1
    res = cv2.warpAffine(img, M, (cols, rows))           # 旋转/位置变换，所用函数一样，只是M不一样
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(res)

"""perspective变换(透视)，Affine变换(仿射)：思路一样，取3-4个点获得变换矩阵M，然后使用同意命令warpAffine()"""
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations
# 仿射变换是正的变斜的

# 透视变换是斜的变正的


'''-------------------------------------------------------------------------
Q. 图片flip的函数？
- 采用np.flip()作为翻转函数，翻转前后size不变
- axis=0表示沿行变化方向，也就是垂直翻转，axis=1表示沿列变化方向，也就是水平翻转
'''
import numpy as np
from numpy import random
def flip_img():
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
def img_pad():
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
def img_rect():
    cv2.rectangle(img, left_top, right_bottom, box_color, thickness)

    cv2.putText(img, label_text, )



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
def wrong_op():
    img = cv2.imread('repo/test.jpg')[:,:,::-1]  #bgr2rgb, 浅拷贝
    #拷贝img至img_copy
    img_copy = img.copy()
    #输入要画的框
    box = np.array([0, 12, 13, 18, 2, 20, 3, 40])
    #画框
    cv2.polylines(img_copy[:, :, ::-1], box.astype(np.int32).reshape(-1,1,2),
                  isClosed=True, color=(255,255,0), thickness=2)
# --------------------正确实例：-------------------------------
def right_op():
    img = cv2.imread('repo/test.jpg')
    #拷贝img至img_copy
    img = img[:,:,::-1].copy()
    #输入要画的框
    box = np.array([0, 12, 13, 18, 2, 20, 3, 40])
    #画框
    cv2.polylines(img, box.astype(np.int32).reshape(-1,1,2),
                  isClosed=True, color=(255,255,0), thickness=2)


'''---------------------------------------------------------------------
Q. 什么是图像的mask，怎么创建mask并使用mask在图像上？
1. 图像的按位操作bitwise operation：cv2.bitwise_not(), cv2.bitwise_and(), cv2.bitwise_or()
2. mask的概念类似pcb板的掩膜概念，用来提取感兴趣的，遮挡不感兴趣的部分
    >
'''
def rect_mask():
    # 创建规则形状的mask
    mask = np.zeros((h,w),dtype=np.uint8)  # 注意mask的数值格式需要根图片一致，所以需要指定成np.uint8，否则后边bitwise操作报错
    mask[100:180,150:300] = 255            # 自定义roi尺寸大小

def bitwise_mask():
    # 创建不规则形状的mask
    img = cv2.imread('test/test_data/opencv_logo.png')
    _, mask = cv2.threshold(bgr2gray(img), 10, 255, cv2.THRESH_BINARY)  # mask，roi区域取255用于保留roi原图
    mask_inv = cv2.bitwise_not(mask)                                    # mask_inv，非roi区域取255用于提取非roi原图

    plt.subplot(121), plt.imshow(mask, cmap='gray')
    plt.subplot(122), plt.imshow(mask_inv, cmap='gray')


'''-----------------------------------------------------------------------
Q. 什么是图像的thresholding？
1. thresholding就是阈值，如果大于阈值则指定为某值(比如255)，小于阈值则指定为某值(比如0)
   也叫做把图像二值化，可用来创建一个二值mask，或者用来
2. 二值化的取值策略：cv2.THRESH_BINARY代表0/255
2. 使用阈值函数cv2.threshold(img, thresh, maxval, type)需要采用灰度图
3. 
'''
def mask_thr():
    # 用阈值函数把图像二值化
    img = cv2.imread('test/test_data/gradient.jpg',0)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    plt.subplot(1,3,1), plt.imshow(img,cmap='gray'), plt.title('original')
    plt.subplot(1,3,2), plt.imshow(thresh1,cmap='gray'), plt.title('binary')
    plt.subplot(1,3,3), plt.hist(img, 255, [0,255])

def mask_thr2():
    # 用阈值函数
    img = cv2.imread('test/test_data/messi.jpg',0)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    plt.subplot(1,3,1), plt.imshow(img,cmap='gray'), plt.title('original')
    plt.subplot(1,3,2), plt.imshow(thresh1,cmap='gray'), plt.title('binary')
    plt.subplot(1,3,3), plt.hist(img, 255, [0,255])



'''------------------------------------------------------------------------
Q. 如何使用特殊thresholding, 比如Adaptive threshold?
cv2.adaptiveThreshold()
'''





'''------------------------------------------------------------------------
Q. 图片部分ROI的抠图以及组合？
1. 基于mask/mask_inv的4步抠图精华： 抠roi，抠非roi，相加，嵌回
2. 按位操作可用来作为抠图动作：与0相与为0(相当于丢弃)，与255相与灰度值不变(相当于保留)
    img = cv2.bitwise_and(src, des, mask), 通常取src与des相同，即在原图操作
    其中src为源图，需要为灰度图，thresh是阈值，maxval是最大值，
参考：https://blog.csdn.net/weixin_35732969/article/details/83779660
'''
def rois():
    img1 = cv2.imread('test/test_data/messi.jpg',1)
    img2 = cv2.imread('test/test_data/opencv_logo.png',1)

    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols]   # 从原图左上角划定一个小区域作为roi

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)  #小图转成灰度图
    plt.imshow(img2gray, cmap='gray')

    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY) # 小图灰度图创建不规则抠图(10-255之间的灰度保留)，符合要求存为255,不符合存为0
    mask_inv = cv2.bitwise_not(mask)                               # 小图mask的取反操作(0变255, 255变0)
    plt.subplot(121), plt.imshow(mask, cmap='gray')
    plt.subplot(122), plt.imshow(mask_inv, cmap='gray')

    # Now black-out the area of logo in ROI                # 基于mask/mask_inv的4步抠图精华： 抠roi，抠非roi，相加，嵌回
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)     # roi与mask_inv按位相与，圆圈区域为0, 相与也是0 (黑色)
                                                       # roi上面抠出洞   
                                                       # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)       # 与mask相与，圆圈区域为255, 相与为像素(保留)
 
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)             # 得到roi的正确形式      
    plt.subplot(221), plt.imshow(bgr2rgb(roi)), plt.title('1.source roi')
    plt.subplot(222), plt.imshow(bgr2rgb(img1_bg)), plt.title('2.bg')
    plt.subplot(223), plt.imshow(bgr2rgb(img2_fg)), plt.title('3.fg')
    plt.subplot(224), plt.imshow(bgr2rgb(dst)), plt.title('4.merge')

    img1[0:rows, 0:cols ] = dst                # 把roi嵌进原图
    plt.imshow(img1)


'''-----------------------------------------------------------------------
Q. 如何区分低通过滤器和高通过滤器，以及如何用低通滤波器
1. 从灰度分布分析相当于空间域的分析，从图像变化的频率分析相当于频域分析。
   图像变化快，就是频率高，就是高频
1. 低通过滤器(low-pass filter/LPF): 去除变化快的点，留下变化慢的点。多用来去除噪声，模糊化blur/平滑化smooth图片
   >平均值滤波：cv2.blur(img, kernel_size)
   >高斯滤波：cv2.GaussianBlur(img, kernel_size, n)
   >中值滤波：cv2.medianBlur()
2. 高通过滤器(high-pass filter/HPF): 也称梯度过滤器，留下变化快的点，多用来检测边沿
   >sobel方向滤波：强调某一方向的高频分量
   >laplacian滤波：
'''
def filter_demo():
    # 平均值blurring: 核是全1, 再除以元素个数
    img = cv2.imread('test/test_data/opencv_logo.png')
    blur = cv2.blur(img, (5,5))             #kernal尺寸是5x5, 越大的kernel理论上平均化程度越宽
    plt.subplot(1,2,1), plt.imshow(img), plt.title('original')
    plt.subplot(1,2,2), plt.imshow(blur), plt.title('average')

    # 高斯blurring: 
    img = cv2.imread('test/test_data/opencv_logo.png')
    blur = cv2.GaussianBlur(img, (5,5),0)
    plt.subplot(1,2,1), plt.imshow(img), plt.title('original')
    plt.subplot(1,2,2), plt.imshow(blur), plt.title('gausian')

    # median中值blurring: 属于非线性滤波，用核中间位置对应的值作为目标值
    img = cv2.imread('test/test_data/opencv_logo.png')
    median = cv2.medianBlur(img, 5)
    plt.subplot(1,2,1), plt.imshow(img), plt.title('original')
    plt.subplot(1,2,2), plt.imshow(median), plt.title('median')


'''-----------------------------------------------------------------
Q. 如何检测边沿
1. 梯度过滤器(gradient filter)：也叫高通滤波(high-pass filter)，包括sobel/scharr/laplacian
    > laplacian变换
    > sobel变换
'''
def edge_detect():
    img = cv2.imread('test/test_data/sudo.jpg',0)
    # laplacian滤波
    laplacian = cv2.Laplacian(img, cv2.CV_64F)         # 拉普拉斯变换，
    # sobel方向滤波
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1,0,ksize=5)   # sobel对噪声更抗噪，ksize=-1则3x3
    sobely = cv2.Sobel(img, cv2.CV_64F, 0,1,ksize=5)

    plt.subplot(2,2,1), plt.imshow(img, cmap='gray'), plt.title('original')
    plt.subplot(2,2,2), plt.imshow(laplacian, cmap='gray'), plt.title('laplacian')
    plt.subplot(2,2,3), plt.imshow(sobelx, cmap='gray'),plt.title('sobelx')
    plt.subplot(2,2,4), plt.imshow(sobely, cmap='gray'),plt.title('sobely')

if __name__ == "__main__":
    edge_detect()

'''-----------------------------------------------------------------------
Q. 如何检测角点？
1. 图像特征就是用计算机语言描述的图像的区别与别的图像的明显特点，比如角点
2. 角点检测器cv2.cornerHarris()
'''



'''------------------------------------------------------------------------
Q. 如何检测边缘，边沿？ - 原来图像检测的那些技术都是从老的图像算法中来的，比如nms/roi/feature pyrimid...
1. 边缘检测的过程：
    step1: 边缘检测对噪声非常敏感，第一步要去除噪声(采用之前低通滤波器gausian blurring)
    step2: 然后通过sobel对水平和垂直分别进行高通滤波得到两张图片，然后基于2张图片得到边沿梯度和每个像素的方向
           梯度方向总是垂直于边沿，
    step3: 然后进行非极大值抑制，也就是扫描每一个像素，确认每个像素是在梯度方向上近邻的最大值，保留这个local maximum
           如果是local maximum则保留进入下一个stage，否则抑制为0， suppressed to zero
    step4: 最后进行滞后阈值hysteresis thresholding判断，定义maxval和minval,如果小于minval必然不是edge则放弃掉,大于maxval必然是则保留。
           而中间部分的如果跟必然是edge的部分能够连接则保留，不能连接则放弃。
2. cv2.Canny()

'''
def edge_detect2():
    img = cv2.imread('test/test_data/messi.jpg',0)
    edges = cv2.Canny(img, 100, 200)   # 定义的100, 200即为阈值
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.subplot(122), plt.imshow(edges, cmap='gray')

# %%
"""
Q. 边缘检测在车道线案例中的应用？
"""
def roi(img, vertices):
    """提取感兴趣区域
    Args:
        img(array):  (h,w,c) or (h,w)
        vertices(array): ()
    """
    mask = np.zeros_like(img)  # 全黑mask，默认就全部覆盖
    if len(img.shape) > 2:     
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count  
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, [vertices], ignore_mask_color)  # 在全黑mask里边填充一块vertices
    masked_img = cv2.bitwise_and(img, mask)           # 发挥mask的功能：与0相与=0(代表覆盖)，与非0相与=
    return masked_img
    

def lane_line_detector(img):
    """基础视觉检测车道线"""
    img_gray = bgr2gray(img)
    low_thr = 40
    high_thr = 150
    img_canny = cv2.Canny(img_gray, low_thr, high_thr)
#    plt.subplot(121),plt.imshow(img[:,:,[2,1,0]])  # bgr to rgb
#    plt.subplot(122)
    plt.imshow(img_canny, cmap='gray')
    
    h = img_canny.size(0)
    w = img_canny.size(1)
    lb = [0, h]
    rb = [w, h]
    apex = [w / 2, 300]
    vertices = np.array([lb, rb, apex], np.int32)
    img_roi = roi(img_canny, vertices)
    

if __name__ == '__main__':
    test_lane_line = False
    if test_lane_line:
    
        path = './test/test_data/solidWhiteCurve.jpg'
        img = cv2.imread(path)
        lane_line_detector(img)
    
    




# %%
'''
Q. 图像直方图histograms有什么用？
图像直方图用来在灰度图下统计每个点象素大小的分布。
1. cv2.calHist(img, channels, mask, histsize, ranges)
   生成的直方图输出是一个array(256,1),代表每种灰度的像素个数。
参考：https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html#histograms-getting-started
'''
def hist_demo():
    # 先创建一个mask: 感兴趣roi取255,其他区域取0
    img0 = cv2.imread('test/test_data/messi.jpg',1)
    img = bgr2gray(img0)
    plt.imshow(img, cmap='gray')
    h,w = img.shape             # (h,w,c) or (h,w)
    mask = np.zeros((h,w),dtype=np.uint8)
    mask[40:240,50:400] = 255
    masked = cv2.bitwise_and(img,img,mask=mask)  # 容易错的点：做bitwise_and()操作的img/mask数据格式需要相同，所以mask创建要声明为np.uint8

    # 最简单的绘制直方图的方法是用plt.hist(data, bins, range)，这也是opencv里边建议的最简方法
    # bins代表格数，range代表数据范围
    plt.subplot(221), plt.imshow(img, 'gray'), plt.title('original')
    plt.subplot(222), plt.imshow(mask,'gray'), plt.title('mask')
    plt.subplot(223), plt.imshow(masked, 'gray'), plt.title('masked img')
    plt.subplot(224), plt.hist(img.ravel(), 256, [0,255]),plt.hist(masked.ravel(),256,[1,256])
    #(这里做masked的直方图时去掉了range中0这个取值，否则因为mask中太多0影响绘图的直观性)

    # cv2自带的一种hist方法，相对麻烦，需要先求出hist的数据
    hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
    
    plt.subplot(221), plt.imshow(img, 'gray'), plt.title('original')
    plt.subplot(222), plt.imshow(mask,'gray'), plt.title('mask')
    plt.subplot(223), plt.imshow(masked, 'gray'), plt.title('masked img')
    plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask), plt.title('hist')


# %%
"""图像处理到底做什么，有哪些假设，有哪些细分领域？
参考：https://zhuanlan.zhihu.com/p/55747295
1. 图像处理的本质：是基于一定假设条件下的信号重建，所谓重建是指恢复信号的原始信息，比如去噪，内插。
   而假设条件包括，比如去噪通常假设噪声是高斯噪声，而内插通常假设边缘连续性和灰度相关性。
2. 

"""


'''--------------------------------------------------------------------
Q. 视频分析的3个主题
1. meanshift/camshift
2. optical flow
3. background subtraction
'''



'''-----------------------------------------------------------------------
Q. 用opencv自带的脸部识别和眼部识别检测器如何做？
'''

def face_detect():
    import numpy as np
    import cv2

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    img = cv2.imread('test/test_data/children.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
