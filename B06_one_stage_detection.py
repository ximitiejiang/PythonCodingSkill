#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:18:35 2019

@author: ubuntu
"""

# %% 基于RetinaNet重新推演一下one stage detector的过程
"""
one stage detector的演进：
可参考: https://github.com/open-mmlab/mmdetection/blob/pytorch-0.4.1/MODEL_ZOO.md
        https://github.com/hoya012/deep_learning_object_detection 
可参考: https://arxiv.org/pdf/1811.04533.pdf 这篇2019年M2Det的论文详细对比了mAP和fps，
       貌似速度上SSD依然最快，其次RefineDet/M2Det；而精度上M2Det最高，其次CornerNet
2. One stage:
    > (2014)最早有multibox的概念
    > (2016)产生SSD和Yolo：             SSD(fps=29.2@1GPU, mAP=25.7@Vgg16)
    > (2017)产生RetinaNet和yolo v2:     RetinaNet(fps=9.1@8GPU, mAP=35.6@Resnet50)
    > (2018)产生CornerNet:              CornerNet(mAP=42.1)
    > (2019)产生M2Det:                  M2Det(mAP=44.2)
"""


# %%
"""Q. SSD的get anchors是如何计算，跟two stage的有什么区别？
区别在于：
1. base anchors的个数不同：ssd的6层特征图每层每个cell的base anchor个数不同.
   分别是[4,6,6,6,4,4]，而在其他two stage上一般每层每个cell的anchor个数相同，
   有的是3个scale*3个ratio=9个，最近的都是1个scale(取8)*3个ratio=3个
2. base anchors的尺寸不同：ssd的base anchors尺寸定义过程，先要定义出一个尺寸阶梯，
   每一层对应一个阶梯(30,60),(60,111),(111,162),(162,213),(213,264),(264,315)，
   也就保证了，浅层采用小尺寸anchor，深层采用大尺寸anchor，然后按照ratio的不同生成。
   而two stage的逻辑更清晰，就是定义一个base size(用的是下采样比例), 然后指定一个
   scale和3个ratio，即得到3个anchor，每层都是一样逻辑，由于采用下采样比例做base size，
   也是能保证浅层anchor小深层anchor大
   
"""
import numpy as np
def ssd_get_anchors(img_size,
                    valid_size,
                    featmap_sizes,
                    base_scale=(0.2,0.9),
                    strides = [8, 16, 32, 64, 100, 300],
                    ratios=None, 
                    scales=None):
    """组合gen_base_anchors(), grid_anchors(), valid_flags()
    
    ssd的prior box就是先验盒本质上就是base anchor，但其生成逻辑不太一样
    参考：https://www.jianshu.com/p/e13792628bac
    参考：https://blog.csdn.net/qq_36735489/article/details/83653816
    但在mmdetection中，还有一点不太一样就是s1_min/s1_max的计算逻辑
    ssd把base anchor叫做prior bbox也就是先验框，也就是在已有经验下定义的尺寸，过程如下：
    1. 定义sk为每个bbox跟原图的尺寸大小之比，smin,smax=(0.2,0.9)，这个比例值可调，比如
    在voc_300定义成(0.2,0.9),voc_512就是(0.15,0.9),coco_300就是(0.15,0.9),coco_512就是(0.1,0.9)
    2. 对每个特征图初始bbox比例求法：sk = smin + (smax-smin)*(k-1)/(m-1)，这其实就是在smin基础上逐级加step=(smax-smin)/(k-1)
    其中m为总的特征图数，但因为s1用另外公式计算所以这里m取5而不是6, 其中k为第k个特征图。
    用该公式可计算出6个比例(0.2,0.37,0.54,0.71,0.88,1.05)
    3. 把base scale乘以图片输入尺寸，就是base bbox的实际尺寸(60,111,162,213,264,315)
    然后把这个实际分寸分段为(60,111),(111,162),(162,213),(213,264),(264,315)就是特征图k2-k5的bbox尺寸范围
    4. 单独计算特征图k1的bbox尺寸范围：原论文采用的0.5×s1，但在mmdetection是这么处理：
    对voc_300(0.1,0.2)，对voc_512(0.07,0.15), 对coco_300(0.07,0.15), 对coco_512(0.04,0.1)
    所以对voc_300的k1，bbox的实际尺寸范围就是(30,60)，
    所有6个特征图k1-k6范围就是(30,60),(60,111),(111,162),(162,213),(213,264),(264,315)
    5. 接下来生成anchors：
    先要定义每个特征图的cell上anchors ratios=[1,1/2,2],或者ratios=[1,1/2,2,1/3,3]
    也就是有的层有3种ratio，有的层有5种ratio，源码定义是0,4,5层是3种，1,2,3层是5种，也就如下：
    这6层的ratio就是([1,1/2,2],[1,1/2,2,1/3,3],[1,1/2,2,1/3,3],[1,1/2,2,1/3,3],[1,1/2,2],[1,1/2,2])
    再要定义scales，统一定义为2个scale，一个scale=1即小方框边长用min_size, 另一个scale=sqrt(max_size/min_size)也就是大方框边长sqrt(min_size*max_size)
    也就是2个scales[1, sqrt(max_size/min_size)]
    所以理论上生成的anchor个数是ratio数*scales数，为[6,10,10,10,6,6]，但实际上源码只从中取了一部分，
    其中保留了小框和相应的ratios以及唯一一个大框，而大框对应的ratios全都丢弃。
    所以各特征图最终生成的anchor个数是[4,6,6,6,4,4]， 其中4为小框3种ratio加一个大框，6为小框5种ratio加一个大框
    
    grid的过程跟其他是一样的, 6个特征图分别进行grid anchor，生成的all_anchors数量
    应该是5776+2166+600+150+36 = 8732个anchors(8732,4)
        
    Args:
        
    Return:
        all_anchors(list): 代表每个特征图的所有anchors[(5776,4), (2166,4), (600,4), (150,4), (36,4), (4,4)]
        valids(list): 代表每个特征图上每个anchors的标志[(5776,), (2166,), (600, ), (150,), (36, ), (4,)]
    """
    # 1. 生成每张featmaps的base anchors
    smin, smax = base_scale
    step = np.floor(100*(smax - smin) / 4.)/100  # 算步长
    sk = np.arange(0.2,1.2,step)                 # 算基础scales
    min_sizes = [np.ceil(sk[i]*img_size) for i in range(len(sk)-1)]   # 算min_size
    max_sizes = [np.ceil(sk[i+1]*img_size) for i in range(len(sk)-1)] # 算max_size
    min_sizes.insert(0, 0.1*img_size)
    max_sizes.insert(0, 0.2*img_size)
    ratios = [[1,1/2,2],           # 表示不同anchor的h/w比例，这是在ssd算法中固定的一个先验数据
              [1,1/2,2,1/3,3],
              [1,1/2,2,1/3,3],
              [1,1/2,2,1/3,3],
              [1,1/2,2],
              [1,1/2,2]]

    base_anchors = []
    for i in range(len(strides)):
        anchor_base = min_sizes[i]
        scale = [1., np.sqrt(max_sizes[i]/min_sizes[i])]
        ratio = ratios[i]
        ctr = [strides[i]/2, strides[i]/2]
        anchors = gen_base_anchors_mine(anchor_base, ratio, scale, ctr)  # 先足量生成anchors
        anchors = anchors[:(len(ratio)+1)]                               # 然后按照源码提取其中的小框+小框变种+大框(也就是前ratio个数+1)
        base_anchors.append(anchors)
         
    # 2. 网格化anchors
    all_anchors = []
    all_valids = []
    for i in range(len(featmap_sizes)):
        all_anchor = grid_anchors_mine(featmap_sizes[i], strides[i], base_anchors[i])
        all_anchors.append(all_anchor)
    
    # 3. 生成valid flag
        valid_feat_h = min(np.ceil(valid_size/strides[i]), featmap_sizes[i][0])
        valid_feat_w = min(np.ceil(valid_size/strides[i]), featmap_sizes[i][1])
        valid_feat_size = [int(valid_feat_h), int(valid_feat_w)] 
        valids = valid_flags(featmap_sizes[i], valid_feat_size, len(base_anchors[i]))
        all_valids.append(valids)
    
    return all_anchors, all_valids



# %% 
"""Q. SSD的anchor target是如何计算的？跟two stage的faster rcnn有什么区别？
"""
def anchor_target_ssd():
    """one stage ssd的anchor target所做的事情跟tow stage类似
    输入相同：都是anchor_list(m,4), gt_bbox(n,4), valid_list(m,4)
    1. assigner: 对每一个anchor指定身份，基于iou的算法原则，生成assigned_list(m,)，依然是m个anchors
    2. sampler: two stage是采用随机采样获得总和256个正负样本，很大程度上克服了样本不平衡的问题
        而在ssd这块采用pseudo sampler也就是假采样，只是从8000多个anchor中提取的正样本和负样本，
        但由于总和依然是8000多个，没有对样本不平衡问题有改善。
    3. anchor_target: 获得label和label_weight，这里的区别在于
        >two stage的anchor head不需要label, 因为只是做前景背景的一个判断，相当与二分类，
        所以只是把正负样本的label=1,无关样本=0。而在ssd这块anchor head实际是作为bbox head使用，
        所以是需要准确label用于后边的分类损失计算，所以label取的是正确的原始值。
        label_weight这块一样都是把正负样本的weight=1，同时bbox也转化为delta用于后边的回归损失计算，没有区别
        >two stage的anchors是把多张特征图汇总到一起计算的，所以有一步unmap
        而ssd不需要unmap操作
    """
    # part1: assigner: 对所有anchor指定身份
    
    # part2: pseudo sampler: 获得正负样本index
    
    # anchor target generate: 生成weight，转换bbox2delta
    
    
    
# %%
"""Q. SSD的loss计算是如何操作的？跟two stage的loss计算有什么区别？
"""
def loss():
    """SSD在计算loss时主体跟two stage一样，分成cls_loss和reg_loss两部分
    1. cls_loss：选择的也是多分类的交叉熵进行损失计算，区别在于
        >由于存在样本不平衡问题(送进来的正负样本之和为8732, 负样本太多)，所以计算损失后
         并没有进行累加缩减，而是先对loss进行排序，取出其中loss最大的一部分，使正负样本
         的数量比为1：3(可设置，在cfg里是neg_pos_ratio=3)，然后在求正负loss的平均值
         作为cls_loss
    2. reg_loss：
    
    """
