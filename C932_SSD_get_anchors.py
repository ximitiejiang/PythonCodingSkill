# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
def gen_base_anchors_mine(anchor_base, ratios, scales, ctr=None, scale_major=True):
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
    if ctr is not None:
        x_ctr = ctr[0]
        y_ctr = ctr[1]
    else:
        x_ctr = 0.5 * w
        y_ctr = 0.5 * h
    
    base_anchors = torch.zeros(len(ratios)*len(scales),4)   # (n, 4)
    if scale_major: # 以scale为主，先乘以scale再乘以ratios
        for i in range(len(scales)):
            for j in range(len(ratios)):
                h = (anchor_base * scales[i]).float() * torch.sqrt(ratios[j])
                w = (anchor_base * scales[i]).float() * torch.sqrt(1. / ratios[j])
                index = i*len(ratios) + j
                base_anchors[index, 0] = x_ctr - 0.5 * w  # 
                base_anchors[index, 1] = y_ctr - 0.5 * h
                base_anchors[index, 2] = x_ctr + 0.5 * w
                base_anchors[index, 3] = y_ctr + 0.5 * h
    
    return base_anchors.round()

def grid_anchors_mine(featmap_size, stride, base_anchors):
    """基于base anchors把特征图每个网格所对应的原图感受野都放置base anchors
    Args:
        featmap_size(list(float)): (h,wn)
        stride(float): 代表该特征图相对于原图的下采样比例，也就代表每个网格的感受野
                      是多少尺寸的原图网格，比如1个就相当与stride x stride大小的一片原图
        base_anchors(tensor): 
    Return:
        all_anchors(tensor): (n,4), 这里的n就等于特征图网格个数*每个网格的base anchor个数(比如3或9个)
    1. 先计算该特征图对应原图像素大小 = 特征图大小 x 下采样比例
    2. 然后生成网格坐标xx, yy并展平：先得到x, y坐标，再meshgrid思想得到网格xx, yy坐标，再展平
       其中x坐标就是按照采样比例，每隔1个stride取一个坐标
    3. 然后堆叠出[xx,yy,xx,yy]分别叠加到anchor原始坐标[xmin,ymin,xmax,ymax]上去(最难理解，广播原则)
    4. 最终得到特征图上每个网格点上都安放的n个base_anchors
    """
    feat_h, feat_w = featmap_size
    shift_x = torch.arange(0, feat_w) * stride  # 先放大到原图大小 (256)
    shift_y = torch.arange(0, feat_h) * stride  #                 (152)
    shift_xx = shift_x[None,:].repeat((len(shift_y), 1))   # (152,256)
    shift_yy = shift_y[:, None].repeat((1, len(shift_x)))  # (152,256)
    shift_xx = shift_xx.flatten()   # (38912,) 代表了原始图的每个网格点x坐标，用于给x坐标平移
    shift_yy = shift_yy.flatten()   # (38912,) 代表了原始图的每个网格点y坐标，用于给y坐标平移
    shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1) # 堆叠成4列给4个坐标[xmin,ymin,xmax,ymax], (38912,4)
    shifts = shifts.type_as(base_anchors)   # 从int64转换成torch认可的float32
    # anchor坐标(9,4)需要基于网格坐标(38912,4)进行平移：平移后应该是每个网格点有9个anchor
    # 也就是38912个(9,4)，也就等效于anchor(9,4)与每一个网格坐标(1,4)进行相加
    # 需要想到把(38912,4)提取出(1,4)的方式是升维到(38912,1,4)与(9,4)相加
    all_anchors = base_anchors + shifts[:,None,:]   # 利用广播法则(9,4)+(38912,1,4)->(39812,9,4)
    all_anchors = all_anchors.view(-1,4)            # 部分展平到(n,4)得到每个anchors的实际坐标(图像左上角为(0,0)原点)                      
    return all_anchors


def valid_flags(featmap_size, valid_size, num_base_anchors, device='cuda'):
    """创建合法标签，用来确定哪些位置点是合法的
    由于输入图片pad以后的尺寸作为初始尺寸h/w，变换到特征图尺寸fh,fw过程中，pytorch默认用下取整，也可设置ceil_mode=True选择上取整
    带来的问题是比如pad后图片为300, 8倍下采样，则上取整ceil(300/8)=38, 就说明feat至少要38，放大后才能涵盖原图。
    >如果实际feat_h=37，那合法的feat就是37(可能会有部分原图丢失) 
    >如果实际feat_h=39,则合法feat就是38(多余的feat是没有用的)
    此时定义一个valid_flag，就是把合法feat尺寸上每个点标注成1,额外非法feat上的点标注为0
    (在ssd中没有pad,所以valid size = img size，且设置了ceil_mode=True，所以所有点都合法
    而在faster rcnn中，图片处理事先设置了size divisor，确保能够整除，所以也能让所有点合法)
    
    Args:
        featmap_size(list): 代表一组特征图的尺寸列表比如[(38,38), (19,19), (10,10), (5,5), (3,3), (1,1)]
        valid_size(list): 代表合法尺寸，从(ceil(pad_img_h/stride), feat_h)中间取小值，代表跟原图相关的特征点，而不是超出图片边界的特征点。
        num_base_anchors(int): 代表该层featmap的每一个cell放置多少个base_anchors (比如ssd是4-6个，fasterrcnn是3个)
    Return:
        valid(tensor): (k,) 其中k代表该特征层每一个anchor的合法标志，k=feat_h*feat_w*num_base_anchors
    """
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
    valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
    valid_x[:valid_w] = 1
    valid_y[:valid_h] = 1
    valid_xx = valid_x[None,:].repeat(feat_h, 1).flatten()
    valid_yy = valid_y[:,None].repeat(1, feat_w).flatten()

    valid = valid_xx & valid_yy
    valid = valid[:, None].expand(
        valid.size(0), num_base_anchors).contiguous().view(-1)
    # 跟用repeat效果一样
    # valid = valid[:, None].repeat(1, num_base_anchors).contiguous().view(-1)
    return valid


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


def SSD_anchor_target():
    """组合assigner(), sampler()
    1. assigner()的方式跟其他是一样的
    2. 
    
    
    """
    
"""---------------------------验证----------------------------------------""" 
if __name__ == "__main__":
    
    import numpy as np
    # base anchors 数据准备
    img_size = 300
    valid_size = 300  # valid size是经过pad之后的size
    base_scale=(0.2,0.9)
    strides = [8, 16, 32, 64, 100, 300]  # 表示下采样比例，SSD_VGG模型决定
    # grid anchors 数据准备
    featmap_sizes = [(38,38), (19,19), (10,10), (5,5), (3,3), (1,1)]
    # get_anchors 汇总调用
    all_anchors, all_valids = ssd_get_anchors(img_size, valid_size, 
                                               featmap_sizes, 
                                               base_scale=(0.2,0.9),
                                               strides = [8, 16, 32, 64, 100, 300])
    # assigner的数据准备
    
    # sampler的数据准备
    
    
    cls_scores = []
    bbox_preds = []
    for i in range(len(sizes)):
        cls_score = torch.randn(1, num_anchors[i]*21, sizes[i][0], sizes[i][0])  # b,c,h,w
        cls_scores.append(cls_score)
        bbox_pred = torch.randn(1, num_anchors[i]*4, sizes[i][0], sizes[i][0])   # b,c,h,w
        bbox_preds.append(bbox_pred)
    



