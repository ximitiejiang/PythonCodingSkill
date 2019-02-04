#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 09:43:06 2019

@author: ubuntu

这部分主要用来从基本概念角度，通过实际代码实践，深刻理解代码
深度学习本质：是loss函数的求极值和求对应参数的问题，利用的是BP算法完成
所以1求loss(先要得到outputs和labels)，2求梯度，3求参数
这其中主要按深度学习计算过程分四步：1.前向传播(output) -> 2.损失计算(loss) -> 3.反向传播(grad) -> 4.参数更新/梯度清零(w)
以下是具体的几个模块分类：
    1. 概率论
    2. 传统机器学习
    3. 网络基础层
    4. 激活函数
    5. 损失函数
    6. 反向传播
    7. 梯度下降与优化器
    8. 正则化
    9. 网络训练
"""

# %%        概率论
"""什么是先验概率，什么是后验概率？怎么相互转化计算
1. 事件的定义：可按照抽取次数来定义事件：第i次抽取为第Ai次事件；也可按事件类型定义A事件，B事件。
   通常Ai次时间相互不独立，而A,B,C..事件相互独立，也就是概率相互不影响。
2. 后验概率：就是基于事件得到的概率，即条件概率即P(B|A)，在A发生的条件下，B发生的概率叫后验概率，因为B是在A事件之后发生的，所以B的概率就是后验概率
   先验概率：就是不借助具体事件,而通过统计数据得到概率，比如P(A)，P(B)
3. 区别P(AB)和P(B|A)：其中P(AB)是指A,B事件同时发生的概率，样本空间是整个S
   而P(B|A)是指在A发生的基础上B发生的概率，此时样本空间从S变为A，所以P(B|A)=P(AB)/P(A)即AB同时发生的概率除以A的样本空间概率
   所以条件概率的本质是样本空间变化。所以一般条件概率更大，因为相对样本空间变小了。
   直白描述是：P(AB)是A发生同时B发生的概率，P(B|A)是已知A已经发生，求B发生的概率，此时A发生一般造成了对样本数量或样本空间的影响。
4. 条件概率公式和乘法定理
    P(B|A) = P(AB)/P(A)
    P(AB) = P(A)P(B|A)
    P(ABC) = P(A)P(B|A)P(C|AB)
    P(ABCD) = P(A)P(B|A)P(C|AB)P(D|ABC)
5. 全概率公式
    P(A) = P(A|B1)P(B1) + P(A|B2)P(B2) + ...
6. 贝叶斯公式
    P(Bi|A) = P(A|Bi)P(Bi)/P(A)
7. 独立事件下的简化公式
    如果事件A,B独立，则P(AB) = P(A)P(B), P(B|A) = P(B)
"""
"""实例1：一等品1,2,3, 二等品4，拿取2次且不放回模式，求第一次一等品条件下，第二次一等品的概率"""
# 区分P(AB)和P(B|A)：
S = [(1,2),(1,3),(1,4),(2,1),(2,3),(2,4),(3,1),(3,2),(3,4),(4,1),(4,2),(4,3)]  # 样本空间，并假定事件A为第一次拿到一等品，事件B为第二次拿到一等品
A = [(1,2),(1,3),(1,4),(2,1),(2,3),(2,4),(3,1),(3,2),(3,4)]                    # 事件A(第一次拿取到一等品)的样本空间
AB = [(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)]
P(A) = A/S = 9/12 = 3/4
P(AB) = AB/S = 6/12 = 1/2
P(B|A) = P(AB)/P(A) = (1/2)/(3/4) = 2/3  # 用条件概率求解，也可以基于事件AB和事件A的次数求解，因为AB和A都是想对于同一样本空间S

"""实例2：r个红球，t个白球，拿取4次且放回模式，每次还加放同色a个球，求第1,2次红球，3,4次白球的概率"""
# 定义A1,A2,A3,A4为第i次抽取红球的事件，~A1,~A2,~A3,~A4为第i次抽取白球的事件, 用条件概率和乘法公式
P(A1) = r/(r+t)
P(A2) = (r+a)/(r+t+a)
P(A1A2(～A3)(～A4)) = P(A1)P(A2|A1)P(~A3|A1A2)P(~A4|A1A2(~A3))
                    = ???


# %%        概率论
"""什么是随机变量？有哪几种随机变量，有什么用？
1. 随机变量是
2. 两种随机变量
    > 离散型随机变量
    > 连续型随机变量
"""
# 待增加代码

# %%        传统机器学习算法
"""如何把高维变量映射到低维？以及什么是PCA？
"""


# %%
"""深度神经网络到底是如何工作的，为什么能够学到东西并做预测
参考：魏秀参的《解析深度学习...》
1. 深度神经网络的第一条数据主线是图像，图像经过卷积/池化/非线性激活等操作，图像的高级语义特征和位置特征
   就被过滤了出来。过滤出来的高级特征在最后一层会被损失函数用于计算预测值与真实值之间的损失。
   然后就基于损失作为输入进入第二条主线，通过反向传播算法，依赖损失函数最小化逻辑，来多轮逐层更新每层参数，
   如此反复的在两条主线来回更新，直到模型收敛，损失达到足够小，从而参数固化达到模型训练的目的。
2. 具体到这三步
    > 主线一：前馈运算，主体是图像，利用的是卷积的特征提取能力
    > 转折点：损失函数，利用他来计算特征预测与真实标签之间的误差作为损失值
    > 主线二：反馈运算，主体是损失，利用损失值来计算更新各个参数
"""




# %%        网络基础层
"""卷积的作用?
参考：https://blog.csdn.net/zouxy09/article/details/49080029
1. 卷积操作作用：卷积在cv领域也叫模板，是已知卷积参数对图像进行过滤计算，得到指定特征。说明了卷积参数能够代表对应特征
   而在神经网络领域是已知特征，通过卷积操作后的损失函数进行梯度下降，得到卷积参数，这个参数就代表了卷积核学到的特征
   由于存在各种各样的卷积核，可以学习到图像的边缘/颜色/形状/纹理等各种模式，并通过组合各种模式会得到更高语义的特征，
   而神经网络会对每一个通道(开始的3层，到64层，到128层，层数越来越多，目的就是增加越来越多的卷积核)分别卷积
2. 卷积核的特点：
   卷积核所有元素之和一般等于1，这样能保证卷积前后图片亮度相同，如果大于1则图片变亮，反之变暗
   卷积后可能会出现负数或者大于255的数，这时需要截断操作到0-255
3. 特定的卷积核能够对图形进行过滤，得到比如水平边缘线/竖直边缘线/整个边缘线等
4. 水平滤波之所以能够找到水平线，是因为在竖直方向上原像素位置取正其他取-1，相当离散版的求导：分别取像素竖直方向前后2点的差，相当与斜率。
   垂直滤波的逻辑也是一样的，只要保证和为0(黑化非边沿像素)，求导方向与线方向垂直(垂直于线方向就是像素变化最大方向，也就是梯度方向)
5. 平均值滤波一方面可以模糊化图片，同时进行局部平均值滤波有去除噪声的效果：因为噪声被认为是零平均值的随机变量，局部平均后噪声就被置0了
6. opencv自带卷积操作函数des=cv2.filter2D(src, -1, kernel)，但该函数只支持单通道所以需要先img的bgr分离或者用灰度图
7. 神经网络的卷积操作，就是用来卷积参数来学习图像特征。
    > conv = nn.Conv2d(in_c, out_c, k_size, stride=1, padding=0, dilation=1, bias=True)，这是默认参数的值
    > 输出层数：in_c自由定义
    > 输出尺寸：out_c计算得到，pytorch原始公式是Hout = (Hin - d*(k_size - 1) + 2p -1)/s + 1 
                                                   = (Hin - k_size + 2p)/s + 1  (这个等效是基于dilation=1得到的)
    > 卷积参数：默认是s=1,p=0但这样会导致图像尺寸缩小，所以通常用s=1,p=1这样能保证图像尺寸不变
    > dilation/bias当前一般用默认参数, 也就相当那个与d=1,b=1（dilation负责卷积的时候扩大范围，bias负责是否增加偏置参数）
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('test/test_data/sudo.jpg', 0)
# 常见卷积核
conv_median = (1/9)*np.array([[ 1, 1, 1],   #这个是中值滤波：亮度相同, 取平均相当与像素点平均化，产生模糊化效果
                              [ 1, 1, 1],   # 
                              [ 1, 1, 1]])
conv_sharp = np.array([[-1,-1,-1],          #这个是锐化滤波：锐化类似边缘检测，把水平垂直倾斜的线条都识别出来，然后+1，则图片亮度不变但会更锐化
                       [-1, 10,-1],          # 卷积核越大效果越明显
                       [-1,-1,-1]])
conv_edge_all = np.array([[0,-4, 0],          #这个是整体边缘检测，类似锐化滤波，差别是总和设为0，则图片变暗突出线条
                          [-4,16,-4],          # 卷积核越大效果越明显
                          [0,-4, 0]])
conv_edge_x = np.array([[0,0,0,0,0],          #这个是水平边沿滤波：变暗，突出边沿线
                      [0,0,0,0,0],            
                      [-1,-1,4,-1,-1],
                      [0,0,0,0,0],
                      [0,0,0,0,0]])
conv_edge_y = np.array([[0,0,-1,0,0],          #这个是水平边沿滤波：变暗，突出边沿线
                       [0,0,-1,0,0],
                       [0,0, 4,0,0],
                       [0,0,-1,0,0],
                       [0,0,-1,0,0]])
conv_moving = (1/5)*np.array([[1,0,0,0,0],          #这个是运动模糊滤波：沿着45方向取同一直线的像素做平均，效果类似图像沿45度移动
                             [0,1,0,0,0],           # 卷积核越大，影响的像素越多，效果越明显
                             [0,0,1,0,0],
                             [0,0,0,1,0],
                             [0,0,0,0,1]])
conv_big_moving = (1/10)*np.eye(10)

# 卷积核的基本运算逻辑：
# 单层卷积核过程简单就是对应位置相乘后累加，
# 多层卷积核过程：？？？
data1 = np.eye(5,5)
conv = np.array([[-1,-1,-1],          
                 [-1, 10,-1],          
                 [-1,-1,-1]])
res1 = cv2.filter2D(data1, -1, conv_sharp)  # 在filter2D()函数中默认是添加padding(0)保证输出与输入的尺寸一样
                                            # 手算第一个数：10*1+(-1)=9 但函数算出来第一个数是6，不过中间的数算出来是对的。
                                            # 似乎filter2D函数的padding策略不同，导致边沿的数算出来不太一样。
data2 = []

# 卷积核用在真实图片的效果：可以看到卷积核能够对图像进行过滤
res2 = cv2.filter2D(img, -1, conv_edge_x)
res3 = cv2.filter2D(img, -1, conv_edge_y)
res4 = cv2.filter2D(img, -1, conv_median)
res5 = cv2.filter2D(img, -1, conv_moving)
res6 = cv2.filter2D(img, -1, conv_big_moving)
res7 = cv2.filter2D(img, -1, conv_sharp)
res8 = cv2.filter2D(img, -1, conv_edge_all)

plt.subplot(421), plt.imshow(img, cmap='gray'),plt.title('original')
plt.subplot(422), plt.imshow(res2, cmap='gray'),plt.title('conv x')
plt.subplot(423), plt.imshow(res3, cmap='gray'),plt.title('conv y')
plt.subplot(424), plt.imshow(res4, cmap='gray'),plt.title('median')
plt.subplot(425), plt.imshow(res5, cmap='gray'),plt.title('moving')
plt.subplot(426), plt.imshow(res6, cmap='gray'),plt.title('big moving')
plt.subplot(427), plt.imshow(res7, cmap='gray'),plt.title('sharp')
plt.subplot(428), plt.imshow(res8, cmap='gray'),plt.title('edge_all')

# pytorch的卷积核操作基本参数：
import torch
import torch.nn as nn
from numpy import random
# 用conv默认参数：s=1,p=0,d=1,b=True，默认参数的问题是无法保持图形尺寸不变
input = torch.tensor(random.uniform(-1,1,size=(8,16,50,100)).astype(np.float32))    # input = (8,16,50,100) b,c,h,w
conv = nn.Conv2d(16, 33, 3, stride=1, padding=0, dilation=1, bias=True)             
output = conv(input)                                                                # output = (8,33,48,98) b,c,h,w， 其中h=(50-3)/1 + 1=48
# 修改默认参数：s=1,p=1，这样能够保证输出图形尺寸不变
input = torch.tensor(random.uniform(-1,1,size=(8,16,50,100)).astype(np.float32))    # input = (8,16,50,100) b,c,h,w
conv = nn.Conv2d(16, 33, 3, 1, 1)                                                   
output = conv(input)                                                                # output = (8,33,50,100) b,c,h,w， 其中h=(50-3+2)/1 + 1=50



# %%        网络基础层
"""下采样和上采样的作用？通常如何实现？
1. 下采样：是指缩小图像，也叫降采样(downsample/subsample)，主要目的是使图像尺寸缩小
    目的是仿照人类视觉系统对图像进行降维和抽象操作。    
    > 下采样之前一般用nn.MaxPool2d(k_size, s=None, p=0, d=1, ceil_mode=False)来定义s=2来实现, ceil模式是指计算输出形状的取整方式是上取整ceil还是下取整floor,默认是floor
                 下采样池化的Hout = (Hin - k_size + 2p -1)/s + 1，公式对卷积与池化层是一样的
                 下采样池化没有可学习参数，只有一些超参数，一般只设置k_size以及s=2, p=0来保证降采样，其他沿用(比如VGG)
                 或者设置k_size以及s=2,p=1来保证降采样，其他沿用(比如Resnet)
    > 下采样现在一般用nn.Conv2d(in_c, out_c, k_size, s=1, p=0, d=1, bias=True)定义s=2来实现
                 下采样卷积的Hout = (Hin - k_size + 2p)/s + 1 (该公式等卷积公式一样，都是基于dilation=1等效出来的)
                 用卷积层做下采样有可学习参数，同时超参数设置k_size=1以及s=2, p=0/1都有，bias=False
    > 用MaxPool2d/AvgPool2d做下采样的好处是：
        特征不变性(用最大值/平均值来代表特征，而不是具体位置的数据，一定程度使学习有一定的空间自由度)，
        特征降维(下采样后尺寸减小，参数个数减少)，
        一定程度防止过拟合(...)
    > 改用conv替代maxpool做下采样的好处是：
        统一了CNN网络，全部用卷积网络，包括用conv替代pooling, 用conv替代fc (在resnet体现)
2. 上采样：是指放大图像，也叫图像插值(umsample/interpolate)
    > 上采样在pytorch中使用F.interpolate(input, size=None, scale_factor=None,mode='nearest'),
      该函数可以同时支持上采样或下采样，可把input转换成size大小或者scale_factor大小之一.
      mode只在上采样时可以选择'nearest','linear','bilinear','trilinear','area'
    > pytorch中原来的upsample()函数已经废弃，被interpolate替代
    > pytorch还有Upsample类定义的层：功能跟interpolat一样，不过只能做上采样不能做下采样
      mode的选择上，nearest(所有)/linear(3D only)/bilinear(4D only)/trilinear(5D only)/area
      3D是指向量(b,c,w), 4D是指图片(b,c,h,w)，5D是指点云(b,c,d,h,w)d为深度
      
"""
import torch
import torch.nn as nn
import numpy as np
from numpy import random
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

# 最大池化和平均池化做下采样
random.seed(3)
data = random.uniform(1,50, size=(3,5,5)).astype(np.float32)
input = torch.tensor(data)  # (c,h,w)=3,5,5
mpool = nn.MaxPool2d(2,stride=2,padding=1)  # 默认参数s=None，应该要用s=1or2, p=1or0
apool = nn.AvgPool2d(2,stride=2,padding=1)  # 输出尺寸计算：h = (5-2+2)/2 + 1 = 3
out1 = mpool(input)   # 最大池化,从(3,5,5)到(3,3,3)
out2 = apool(input)   # 平均池化,从(3,5,5)到(3,3,3)

# 用卷积层做下采样
random.seed(3)
data = random.uniform(1,50, size=(2,3,5,5)).astype(np.float32)
input = torch.tensor(data)  # (b,c,h,w)=3,5,5
conv_d = nn.Conv2d(3, 6, 1, stride=2, padding=0, bias=False)   # 输出尺寸计算：当p=0时h = (5-1)/2 + 1 = 3， 当p=1时h=(5-1+2)/2 +1=4
out3 = conv_d(input)   # (2,3,5,5) -> (2,6,3,3)  


# 用interpolate()进行上采样: 注意需要输入(b,c,h,w)才能正常进行上下采样，对w,h同时缩放，如果没有b的一维输入会不正常
random.seed(3)
data = random.uniform(1,50, size=(2,3,5,5)).astype(np.float32)
input = torch.tensor(data)  # (b,c,h,w)=2,3,5,5
out4 = F.interpolate(input, scale_factor=2, mode='nearest') # (2,3,5,5) to (2,3,10,10)


from datasets.transforms import imresize
img = cv2.imread('test/test_data/messi.jpg')
img1 = imresize(img,(300,300))    # 图像先resize到300x300
img2 = torch.tensor(img1.astype(np.float)).permute(2,0,1).unsqueeze(0)  # h,w,c to b,c,h,w
out5 = F.interpolate(img2, scale_factor=2, mode='nearest')  # 从(1,3,300,300) -> (1,3,600,600)。。
img_s = out5.squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
plt.subplot(121), plt.imshow(img1[...,[2,1,0]])    # 
plt.subplot(122), plt.imshow(img_s[...,[2,1,0]])   



# %%        网络基础层
"""BatchNorm层是什么，有什么功能，如何使用？
参考一个比较直观的好的理解：https://www.cnblogs.com/guoyaohua/p/8724433.html
0. nn.BatchNorm2d(n_features, eps=1e05, momentum=0.1)
1. batchnorm也叫批归一化操作，简称BN，是google在2015年提出来的。操作方式类似于对整个数据集进行归一化的操作过程
    先对一个batch的数据计算其均值和方差，然后对batch数据进行规范化(x-mean)/std，最后加入2个可学习参数gamma/beta来计算输出x=gama*x + beta
2. 做BN的原因是数据在神经网络传递过程中，每一层的分布情况会发生变化，叫做internal covariate shift，这种分布变化导致
3. 公式的理解：对一个batch进行归一化比较好理解，就是让数据的分布先统一成标准正态分布N(0-1)，这样尽可能的减少因为数据分布差异导致的梯度消失或者梯度爆炸
   然后加入可学习参数gamma/beta相当于是逆归一化，这一步是避免数据真的都变成标准正态分布后神经网络无法学习到东西的情况，就由神经网络自己学习来决定每个batch的分布。
   比如当gamma/beta正好等于图片mean/std则相当于还原到原来x，BN没起作用，采用的就是数据本来的分布，如果gamma/beta学到=0,就相当于数据为标准正态分布，
   整个过程就保证了每个batch的数据分布情况尽可能一致(不会变化过大导致梯度消失/爆炸)，但又保证了数据分布的多样性(神经网络有东西可以学习)
4. 由于batchnorm层有2个参数gamma/beta，所以需要初始化，默认初始化策略是gamma从U(0,1)取，beta设为0
   不过mmdetection的初始化策略是constant_init(model, 1)，也就是统一设置为1
5. 批归一化和数据集归一化的区别：？
6. batchnorm的局限性：当前的理论并没有很好的解释为什么batchnorm是有效的。
   同时batchnorm的让数据之间的差异变小了(因为分布被限制在了(0,1)附近)，在超分辨率的应用领域不适合，韩国人超分辨率模型就不用batchnorm
   用了batchnorm一般不用dropout了，这里如果batchnorm不适用，可考虑dropout???
"""
from numpy import random
# 很多cnn的经典模块：conv+bn+relu
conv = nn.Conv2d(1, 32, 3, stride=1, padding=1)
bn = nn.BatchNorm2d(32)            # 只需要设置一个输入通道数即可
relu = nn.ReLU(inplace=True)
# 用x.mean(), x.std()考察每一步的分布变化情况        
x1 = random.randn(100,100)*0.5 - 2   # 基于标准正态分布创建一个正态分布: 要从(0,1)变为(-2, 0.5)就是逆运算
x2 = torch.tensor(x1.astype(np.float32)).unsqueeze(0).unsqueeze(0)   # (h,w) to (1,1,h,w)
x3 = conv(x2)                     # 经过卷积以后的分布会发生变化,从(-2,0.5)变成(-0.17,1.3)
x4 = bn(x3)                       # 经过bn以后的分布会调整到(0,0.5)这个数据主要跟BN的初始化有关,但都是靠近分布(0,1)
x5 = relu(x4)                     # bn之后的数据分布在(0,1)，相比于其他分布，这个分布？？？？

from mmcv.cnn import constant_init
constant_init(bn,1)
x5 = bn(x3)                       # 用constant)init()后分布调整到(0,1)附近


# %%        网络基础层
"""全连接层的作用？
参考：https://blog.csdn.net/m0_37407756/article/details/80904580
0. nn.Linear(in_f, out_f, bias=True)
1. 全连接层叫fully connected layers也就是
   他的作用就是把带位置的卷积特征转化为了不带位置的分类特征。
   比如一只猫在卷积特征的左上角，另一只猫在卷积特征的右下角，通过在卷积层的位置以及卷积参数可以得到分类和位置信息。
   但转化成全连接层后，数据被串联起来并整合，这样位置信息就丢掉了，而分类信息的特征被整合成一组数，每个数值代表一个分类的概率。
   因此全连接层不适合用来做object detection或者segmentation，只适合做分类。
2. 全连接层的缺点：一方面是参数冗余，大约80%的网络参数是由全连接层产生的，另一方面是丢弃了位置信息，只保留了分类信息，不能做检测和分图
3. 全连接计算过程：
    先用view()或reshape()把特征图数据拉直成(b, c*h*w), 其中c*h*w就想当于全连接层神经元个数
    然后通过Linear()变换特征数据的长度，也就是变换神经元个数，经过n轮全连接层后，
    把神经元个数变换为分类数，从而每个神经元代表一个分类的概率值。
    (此时由于是多分类概率值，损失函数需要选择？？？)
    核心理解：全连接层 
4. 用AvgPool代替全连接层：
    nn.AvgPool2d(k_size, s=None, p=0, ceil_mode=False)
    平均池化的操作方式：
5. 用大卷积替代跟卷积相连的全连接层，用1x1小卷积替代跟fc相连的全连接层：
    此时，把大卷积看成把hxw的尺寸缩减到1x1，然后用小卷积1x1进一步调整层数(对应)
    务必把全连接操作理解成2类卷积操作。
    nn.Conv2d()

"""
# 对一个224x224x3的input,经过VGG转换为7x7x512的特征图后，如何经过全连接？

import numpy as np
from numpy import random
import torch.nn as nn
import torch

"""案例2：实现VGG的全连接层"""
x1 = random.uniform(-1,1, size=(512,7,7))  
x2 = torch.tensor(x1.astype(np.float32)).unsqueeze(0)  # 特征图尺寸为(b,c,h,w)
# 以下是VGG的classifier，用全连接层来实现的
l1 = nn.Linear(512 * 7 * 7, 4096)
r1 = nn.ReLU(True)
d1 = nn.Dropout()
l2 = nn.Linear(4096, 4096)
r2 = nn.ReLU(True)
d2 = nn.Dropout()
l3 = nn.Linear(4096, 2)
# 全连接计算：
x3 = x2.view(x2.size(0), x2.size(1)*x2.size(2)*x2.size(3)) # 先把特征拉直成一列神经元(b, c*h*w)
x4 = l1(x3)           # 从(1,25088) -> (1,4096)
x5 = l2(d1(r1(x4)))   # (1,4096) -> (1,4096)
x6 = l3(d2(r2(x5)))   # (1,4096) -> (1,2)

"""用大的平均池化层代替全连接层：resnet的实现方式 (1x512x7x7) to (1,2)
相比于VGG的全连接的变换，这里省略了2个全连接层，且AvgPool是无参的，参数可以减少很多，模型尺寸减小但精度不会下降"""
avg1 = nn.AvgPool2d(7, stride=1)  # 池化核为7x7
fc1 = nn.Linear(512, 20)

x7 = avg1(x2)                     # 用平均池化拉直特征，从(1,512,7,7) to (1,512,1,1)
x8 = x7.view(x7.size(0),-1)       # 去除多余维度，(1,512,1,1) to (1, 512)
x9 = fc1(x8)                      # 变换到分类概率，转换为(1,20)

"""用大卷积+小卷积层替代全连接层的方法：确保p=0的设置
方法：用大卷积替代跟卷积相连的全连接层，用1x1小卷积替代跟fc相连的全连接层
这种方式来自魏秀参的知乎https://www.zhihu.com/question/41037974/answer/150585634"""
c1 = nn.Conv2d(512, 4096, 7, stride=1, padding=0)  # h=(7-7+2*0)/1 + 1 = 1, 注意p=0才能保证输出尺寸=1 
c2 = nn.Conv2d(4096,2048, 1, stride=1, padding=0)
c3 = nn.Conv2d(2048,20, 1, stride=1, padding=0)
x10 = c1(x2)               # 用大卷积得到(1,4096,1,1)
x11 = c2(x10)              # 用小卷积缩小进一步提炼缩小特征
x12 = c3(x11)              # 用小卷积缩小进一步提炼缩小特征到分类数
x13 = x12.reshape(1,-1)     # 去除维度为1的部分，得到(b,c)即等效为fc的输出了

# %%        网络基础层
"""Dropout层的作用是什么，在哪些地方使用"""



# %%        网络基础层
"""为什么当前所有神经网络都是设计成很多基础层叠加，形成很深的神经网络，并且特征尺寸越来越小，特征通道越来越多？
1. 多层神经网络的优点，一个核心就是可以有扩展的感受野
2. 感受野：是指后一层一个节点在前一层或前n层所对应的特征尺寸大小，就是这个节点在前n层的感受野大小
   比如一个7x7卷积核的后一层单个节点对应前一层感受野大小就是7x7，而如果是3x3卷积核往后3层的节点也能在前3层得到同样7x7的感受野(节点-3x3-5x5-7x7)
3. 虽然小卷积核与大卷积核都能得到相同感受野，但小卷积核有很多优势：
    > 小卷积核需要多层叠加形成大的感受野，多层能够
4. 在同一层神经网络输出的多层特征图中：
5. 在不同深度的不同特征图中：
"""





# %%        激活函数
"""解释不同激活函数的优缺点？
1. 激活函数的功能：也叫非线性映射函数，
1. 常见激活函数
    > sigmoid()
    > relu()
    >
2. 
"""
# 常见激活函数曲线
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 如果没有激活函数，无论多少层都只是线性变换

# 常见激活函数
x = np.arange(-5 ,5, 0.1, dtype=np.float32)
x_t = torch.tensor(x)

y_sigmoid = F.sigmoid(x_t).numpy()
y_relu = F.relu(x_t).numpy()
y_tanh = F.tanh(x_t).numpy()

plt.subplot(231), plt.plot(x, y_sigmoid), plt.title('sigmoid')
plt.subplot(232), plt.plot(x, y_relu), plt.title('relu')
plt.subplot(233), plt.plot(x, y_tanh), plt.title('tanh')

# relu的优点，sigmoid的缺点

# %%        损失函数
"""损失函数的定义是什么？为什么要有损失函数？
1. 损失函数：损失函数本质上就是求极值函数，也就是反向传播算法的目标函数。
   是指所有样本的预测值与标签值之间的偏差函数。这种偏差通常采用不同的方式评估。
   比如通过欧式距离最小来评估，比如通过交叉熵最小来评估
   损失函数是关于一组参数(w1..wn)的函数，为了预测最精确就是让损失函数最小，所以通过求解损失函数的最小值来得到这组参数。
2. 逻辑回归的过程：一组数据(x1,..xn)经过线性映射到h(w) =w1x1+w2x2..+wnxn, 再经过非线性映射g(theta)=sigmoid(h)
   这样就建立(x1..xn) -> h(w1..wn) --> g(theta) 的映射，我们认为g就是预测值，损失函数采用欧式距离评估
   因此通过找到损失函数最小值，来得到最小值时的参数(w1..wn)
3. 如何选择损失函数：
   nn.LossFunc是基于module的损失函数类
   nn.functional.lossfunc是函数形式的损失函数，一般使用函数形式更简单
   
"""
# 尝试简单计算下各个损失函数
import torch
import torch.nn.functional as F
import numpy as np

d1 = torch.tensor([[1.,1.], [1.,1.]])
d2 = torch.tensor([[0.,1.], [2.,3.]])

"""nn.MSELoss/F.mse_loss: mean squared error 均方误差损失: 每个对应元素差的平方，默认平均缩减，也可用求和缩减
   F.mse_loss(d1,d2,reduction='mean')
   只需控制reduction, 另两个参数size_average/reduce已废弃
   默认的loss输出是缩减操作以后的输出平均值mean，可以选择其他缩减方式reduction='none','batchmean','sum','mean'这4种"""
loss1 = F.mse_loss(d1, d2)                   # 默认取平方和的均值
loss2 = F.mse_loss(d1, d2, reduction='sum')  # 取平方和

"""nn./F.crossentropy 交叉熵误差: 
   F.crossentropy(d1,d2,reduction='')"""
loss1 = F.cross_entropy(d1, d2)   # 

"""nn.LogSoftmax/F.log_softmax"""


"""nn.NLLLoss/F.Nll_loss"""

# 损失函数之2：交叉熵


# %%        损失函数
"""熵的概念和计算，以及交叉熵概念，以及交叉熵作为损失函数的意义？
1. 信息熵概念：是指某一个离散型随机变量X的不确定性的度量，随机变量X的概率分布为p(X)=pk (k=1..n) 
    >Ent(X)= - sum(pk*log(pk)), 其中Ent(X)就是系统的信息熵，pk代表每一个事件的概率
     从公式可看出信息熵支持多分类标签的计算(而基尼指数公式就只支持二分类)
    >每一个数据集看成一个系统就有一个信息熵，每一个子集也有一个信息熵，都代表了这个系统基于这个分类方式的混乱程度
    >
2. 信息增益：是指系统基于某种特征划分下的信息熵增益
    >gain = Ent(D) - sum((Di/D)*Ent(Di)), 其中Ent(D)为集合的信息熵，而(Di/D)*Ent(Di)为某一子集合的条件熵
     可理解为集合在加入某特征后不确定度减小的程度，也就是增益。

3. 交叉熵：是
    >
4. 相对熵：也叫KL散度
"""
def calEntropy(data, n):
    """计算一个数据集作为随机变量的信息熵, data为mxn array, n为第n列作为随机变量的概率事件"""
    from math import log
    numEntries = len(data)
    labelcounts = {}  # 字典用于放置{总计多少类：每个类多少样本个数}
    for line in data: # 循环取每一行样本
        currentlabel = line[-1]
        if currentlabel not in labelcounts.keys(): # 如果是一个新分类
            labelcounts[currentlabel] = 0       # 在字典新增这个新分类，对应样本个数0
        labelcounts[currentlabel] +=1    # 如果不是新分类： 在旧类的样本个数+1 
    # 计算一个数据集的信息熵
    shannonEnt = 0.0
    for key in labelcounts:
        pi = float(labelcounts[key])/numEntries # 计算每一个分类的概率p=分类数/总数
        shannonEnt -= pi * log(pi,2)    # 计算entropy = -sum(p*logp)
    return shannonEnt

def calCrossEntropy():
    pass

def test():
    # 读取数据
    persons = []
    cols = ['age', 'has_job', 'has_house','has_loan', 'approved']  # 用随机变量来理解数据：随机变量就是贷款能否获得批准的事件的概率
    with open('test/test_data/loan.txt') as f:
        loans = f.readlines()
        for loan in loans:
            loan = loan.split()
            persons.append(loan)
    e0 = calEntropy(persons)
    print('total entropy: {:.6f}'.format(e0))



# %%        反向传播
"""CNN的算法原理, BP相关理论的python如何实现?
参考：http://jermmy.xyz/2017/06/25/2017-6-25-reading-notes-neuralnetworksanddeeplearning-2/
参考：http://jermmy.xyz/2017/12/16/2017-12-16-cnn-back-propagation/  (这个是从全连接BP谈起，到卷积BP)
参考：http://www.cnblogs.com/pinard/p/6494810.html (刘建平博士的完整推导)
参考：https://www.zhihu.com/question/27239198?rf=24827633 (有各种对反向传播的理解方式，但我最欣赏Jermmy的博客介绍)
参考：https://blog.csdn.net/g11d111/article/details/83021651 (介绍pytorch如何源码实现的，很难理解)
参考：http://www.cnblogs.com/yjphhw/p/9681773.html (选用了里边的2个实例)
核心要区别开来反向传播算法(BP/backpropagation)和复合函数求导算法
1. 反向传播的核心是：找到极值函数(比如loss)，然后对极值函数进行反向传播，计算梯度，更新参数
   先根据预测和标签计算出来loss，然后基于loss反向传播计算梯度
   理论上可以计算出输出loss对每一个参数w的梯度，但这样会有很大的冗余计算，因为每一个梯度计算
   都是从每个参数w的叶子节点跑到输出点，中间肯定很多重复计算的中间节点。所以反向传播
   算法跟常规复合函数求导的差别在于，他是从输出点逐步把计算往回算，这样任何一个针对w的
   梯度都只需要前一层的数据就能算出来，也就不存在重复计算了。
2. 推导过程：
    最简单的单线网络的反向传播
    次简单的并线网络的反向传播
    卷积网络的反向传播
3. 具体计算步骤：
    step1: 前向传播，outputs = forward(inputs),逐层计算输出outputs
    step2: 计算损失，loss = criterion(outputs, labels)
    step3: 反向传播，loss.backward(), 其中loss为tensor, 根据BP反向传播算法，
           基于损失值先计算每个神经元(特征图上像素点)的残差，然后计算损失对每个参数的梯度
           查看方式：parameters.grad???
    step4：参数更新，optimizer.step(), 基于梯度更新每个参数值
           梯度清零，optimizer.zero_grad(), 在参数更新后梯度就可以清零了，该步也可以放在每个iter的最开始
           查看方式：parameters
"""
""" 实例1，实现一个最简单只有一层隐含层的网络，脱离pytorch/tensor，完全python实现底层
    实现简单的前向传播，损失函数，反向传播
    实验基本的多轮训练，学习率改变"""
import numpy as np
from math import exp
import matplotlib.pyplot as plt
inputs = np.array([[0.35],[0.9]])
labels = np.array([0.5])
w0 = np.array([[0.1,0.8],[0.4,0.6]])
w1 = np.array([0.3,0.9])

def sigmoid(x):
    """x is ndarray, returns ndarray"""
    return 1/(1+np.exp(-x))

def derive_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def mse_loss(inputs, labels):
    return 0.5*(inputs-labels)**2

epoch=5000
# 增加对学习率的设置
lr = 0.01
ep = []
lo = []          
for k in range(epoch):
    # 前向传播
    z1 = np.dot(w0, inputs)        # 前向传播：第一层w0*x
    a1 = sigmoid(z1)               # 前向传播：第一层激活函数f(w0*x)
    z2 = np.dot(w1, a1)            # 前向传播：第二层w1*x
    a2 = sigmoid(z2)               # 前向传播：第二层激活函数f(w1*x)
    # 反向传播
    loss = mse_loss(a2,labels)     # 计算损失函数，输出损失值
    if k%10==0:
        print('epoch: %d, loss: %.6f, predict value: %.4f'%(k,loss, a2))
        ep.append(k)
        lo.append(loss)
    delta2 = 1 * derive_sigmoid(z2)          # 反向传播：求解delta作为残差反向传播
    delta1 = delta2 * w1 * derive_sigmoid(z1) 
    dw0 = delta1 * inputs                    # 反向传播：求每个参数的梯度
    dw1 = delta2 * a1                        
    dw1 = dw1.flatten()
    # 优化器更新参数
    w0 -= dw0*lr                             # 参数更新与优化：更新每个参数
    w1 -= dw1*lr
    # 为了顺利训练，进一步设置可变学习率
    if k > 200:
        lr = 0.001
    if k > 2000:
        lr = 0.0001
plt.plot(ep, lo)  # 打印损失曲线

"""实例2: 活用反向传播，计算函数极值"""
# 计算函数极值y = x^2 + 2x + 1: 函数为f(x), x为参数， 类似于loss为w1..wn的函数所以对loss求导
# 初始化参数
x = np.array([0],dtype=np.float32)
x = torch.tensor(x, requires_grad=True)
lr = 0.01
epoch = 300
record_y = []
record_x = []
# 训练
for i in range(epoch):
    # 函数值的前向计算
    y = x**2 + 2*x + 1
    # 函数反向传播：求梯度
    y.backward()               # 反向传播：计算误差，反向传播，更新梯度
    # 更新参数
    x.data -= x.grad.data*lr   # 梯度下降更新参数
    x.grad.data.zero_()        # 梯度清零
    record_x.append(x.detach().numpy().copy())  # 梯度更新的变量不能直接转numpy，需要先detach再转numpy
    record_y.append(y.detach().numpy())         # detach的变量不再更新梯度，但依然指向同一tensor
    print('current min y=%f.2, x=%f.2'%(y,x))
plt.subplot(121),plt.plot(record_x),plt.title('x')
plt.subplot(122),plt.plot(record_y),plt.title('y')

"""实例3: 活用反向传播，计算拟合函数的参数"""
# 计算一组数据的拟合函数，得到拟合参数
from numpy import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
n = 100
x0 = random.rand(n)               # x为随机平均分布的n个数
y0 = 3*x0 + 1 + random.rand(n)/3  # y为3x + 1 + 随机噪声
# 目标拟合 y = kx + b
x = torch.tensor(x0.astype(np.float32), requires_grad=True)
y = torch.tensor(y0.astype(np.float32), requires_grad=True)

# 初始化参数
k = torch.tensor([1.], requires_grad=True)  
b = torch.tensor([0.], requires_grad=True)
epoch = 1000
lr = 0.001
record_loss = []
record_epoch = []
for i in range(epoch):
    # 计算损失
    loss = F.mse_loss(k*x+b, y)  # loss函数评价的是拟合函数值与目标值的误差
    # 反向传播：计算误差，计算梯度
    record_loss.append(loss.detach().numpy())
    record_epoch.append(i)
    loss.backward()   
    # 梯度下降：更新参数，采用简单的SGD算法w = w - grad*lr
    # 这里没有采用pytorch自带的优化器，所以优化所做的工作需要手动完成：提取梯度，更新参数，梯度清0
    k.data -= k.grad.data * lr
    b.data -= b.grad.data * lr   # 修改requires_grad=True的参数值，需要.data绕开计算图的锁定
    print('loss=%.5f, k=%f.3, d=%f.3'%(loss,k,b))
    k.grad.data.zero_()   
    b.grad.data.zero_()      # 修改t.grad这个tensor的值，同样需要.data绕开计算图的锁定
plt.subplot(121), plt.plot(record_epoch, record_loss)
plt.subplot(122), plt.scatter(x.detach().numpy(),y.detach().numpy()), plt.plot(x.detach().numpy(), k.item()*x.detach().numpy() + b.item())


# %%        反向传播
"""反向传播本质是为了求出每个参数的梯度，那什么是这个过程的梯度消失/梯度爆炸？如何避免？
为了规避梯度消失，梯度爆炸，有如下方法：
1. 添加Batchnorm层：

2. 添加dropout层：

3. 
"""



# %%        梯度下降与优化器
"""梯度下降做了什么？优化器用来做什么，如何定义合适的优化器
0. 梯度下降算法是整个CNN能够训练并得到最优模型的核心原因，他是指：
   在每个batch计算出梯度以后，按照不同的优化算法，让参数能够沿着梯度下降的方向更新值，
   只要参数是沿着梯度下降的方向(w -= 梯度)进行更新，也就是-dloss/dw方向，就能够保证loss是最快减小的方向
   也就能保证loss最终达到最小值。
   可以通过等高线来理解：假定只有w1,w2两个参数的loss,则w1,w2在平面上形成一个对loss的等高线
   垂直于等高线方向的就是梯度方向，只要沿着垂直等高线方向走，总是最终能够并且最快到达等高线中心点，
   也就是loss的最小值点上。
1. 优化器主要做如下事情：
    a. 进行参数更新：w = w - grad*lr (基于参数/参数的梯度/学习率/优化算法其他超参数)
    b. 进行梯度清零：t.grad.dat.zero_(), 或者optimizer.zero_grad()
2. 常用优化器的类型，区别：
    a. SGD()
    b. Adam()
"""
from torch.optim imprt SGD
optimizer = SGD()

# 优化器的基本使用逻辑：针对每个iter的mini_batch，计算一次梯度，更新一次参数
# 优化器初始化：传入模型参数列表即可，以及一些优化的超参数，特别是初始学习率这个核心参数
for data,label in dataset:
    optimizer.zero_grad()       # 每个iter的mini_batch作为独立部分，单独计算损失和梯度，用于更新参数
    output = model(data)        # 前向传播算法：计算输出
    loss = F.mse(output, label) # 损失函数：计算损失
    loss.backward()             # 反向传播算法：计算误差，误差反向传播，每个参数的梯度更新
    optimizer.step()            # 优化器算法：更新每个参数(基于参数列表，其梯度可从参数属性直接获取)

# 优化器的类型：
#   Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#   SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0)



# %%        梯度下降与优化器
"""凸函数概念，求极值方法，为什么用梯度下降而不是牛顿法?
梯度下降优化算法分2类，第1类是一阶梯度下降算法，第2类是二阶梯度下降算法
1. 大部分情况都是采用一阶梯度下降算法
2. 二阶梯度下降算法缺点：
"""




# %%        正则化
"""解释l0/l1/l2正则化，以及如何在深度学习网络中使用？
"""




# %%        网络训练
"""过拟合与欠拟合的概念，原因，解决办法？
"""


# %%        网络训练
"""网络参数初始化的方法有哪些，如何选择初始化方法
1. conv2d使用kaiming_init(model)
2. Batchnorm使用constant_init(model, val)

"""


# %%        网络训练
"""网络超参数有哪些，如何定义超参数？
1. 模型各层参数：
    > Conv2d()
    > MaxPooling2d()
"""



# %%        网络训练
"""样本不平衡问题是什么，影响是什么，怎么解决样本不平衡问题？
"""



