# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
def gen_base_anchors_mine(anchor_base, ratios, scales, scale_major=True):
    """生成n个base anchors/[xmin,ymin,xmax,ymax],生成的base anchors的个数取决于输入
    的scales/ratios的个数，早期一般输入3个scale和3个ratio,则每个网格包含9个base anchors
    现在一些算法为了减少计算量往往只输入一个scale=8, 而ratios输入3个[0.5, 1.0, 2.0]，
    所以对每个网格就包含3个base anchors
    生成的base anchor大小取决与anchor base大小，由于每个特征图的anchor base都不同，
    所以每个特征图对应base anchor大小也不同，浅层大特征图由于stride小，对应anchor base
    也小，也就是大特征图反而对应小anchor，数量更多的小anchor
    Args:
        anchor_base(float): 表示anchor的基础尺寸
        ratios(list(float)): 表示h/w，由于r=h/w, 所以可令h'=sqrt(r), w'=1/sqrt(r), h/w就可以等于r了
        scales(list(float)): 表示整体缩放倍数
        scale_major(bool): 表示是否以scale作为anchor变化主体，如果是则先乘scale再乘ratio
    Returns:
        base_anchors(tensor): (m,4)
    1. 计算h, w和anchor中心点坐标(是相对于图像左上角的(0,0)点的相对坐标，也就是假设anchor都是在图像左上角
       后续再通过平移移动到整个图像每一个网格点)
        h = base * scale * sqrt(ratio)
        w = base * scale * sqrt(1/ratio)
        x_ctr = h/2
        y_ctr = w/2
    2. 计算坐标
        xmin = x_center - w/2
        ymin = y_center - h/2
        xmax = x_center + w/2
        ymax = y_center + h/2
        
    """
    ratios = torch.tensor(ratios) 
    scales = torch.tensor(scales)
    w = anchor_base
    h = anchor_base
    x_ctr = 0.5 * w
    y_ctr = 0.5 * h
    
    base_anchors = torch.zeros(len(ratios)*len(scales),4)   # (n, 4)
    for i in range(len(ratios)):
        for j in range(len(scales)):
            h = anchor_base * scales[j] * torch.sqrt(ratios[i])
            w = anchor_base * scales[j] * torch.sqrt(1. / ratios[i])
            index = i*len(scales) + j
            base_anchors[index, 0] = x_ctr - 0.5 * w  # 
            base_anchors[index, 1] = y_ctr - 0.5 * h
            base_anchors[index, 2] = x_ctr + 0.5 * w
            base_anchors[index, 3] = y_ctr + 0.5 * h
    
    return base_anchors.round()


def ssd_get_anchors(anchor_bases, ratios, scales, scale_major=True,
                    featmap_sizes, stride):
    """组合gen_base_anchors()和grid_anchors()来生成all anchors
    ssd的base anchor生成逻辑不太一样，参考：https://www.jianshu.com/p/e13792628bac
    ssd把base anchor叫做prior bbox也就是先验框，也就是在已有经验下定义的尺寸
    先定义了min_size = 30, max_size=60，base_ratio_range=[0.2,0.9]
    scale = smin + (smax-smin)*(k-1)/(m-1)其中m为
    0. base_ratio_range跟数据集有关：coco_[0.15, 0.9], voc[0.2,0.9]
    1. 每个cell的anchor个数定义：SSD的论文已定义为[4,6,6,6,6,4]
    2. 每个base anchor尺寸定义：先定义了一个, 并只定义一个ratio=h/w=1/2
       然后先以min_size生成一个，
        
    Args:
        
    Return:
        
    """
    # 1. 生成每张featmaps的base anchors
    base_anchors = []
    for anchor_base in anchor_bases:
        base_anchor = gen_base_anchors_mine(anchor_base, ratios, scales)
        base_anchors.append(base_anchor)
    # 2. 网格化anchors
    all_anchors = []
    for i in range(len(featmap_sizes)):
        all_anchor = grid_anchors_mine(featmap_sizes[i], stride[i], base_anchors[i])
        all_anchors.append(all_anchor)
    # 3. 生成valid flag
    
    
    
"""---------------------------验证----------------------------------------"""    
sizes = [(38,38), (19,19), (10,10), (5,5), (3,3), (1,1)]
num_anchors = [4, 6, 6, 6 ,4, 4]
anchor_strides = [8, 16, 32, 64, 100, 300]  # 用于strides
# 这两组数跟输入img大小相关，计算得到
min_sizes = [30, 60, 111, 162, 213, 264]    # 用于base size
max_sizes = [60, 111, 162, 213, 264, 315]
for i in range(len(anchor_strides)):
    scales = [1., np.sqrt(max_sizes[i]/min_sizes[i])]
    ratios = [1.]
    ctr = ((stride -1 ) / 2., (stride - 1)/2)
    

cls_scores = []
bbox_preds = []
for i in range(len(sizes)):
    cls_score = torch.randn(1, num_anchors[i]*21, sizes[i][0], sizes[i][0])  # b,c,h,w
    cls_scores.append(cls_score)
    bbox_pred = torch.randn(1, num_anchors[i]*4, sizes[i][0], sizes[i][0])   # b,c,h,w
    bbox_preds.append(bbox_pred)
    
gen_base_anchors_mine(anchor_base, ratios, scales, scale_major=True)




