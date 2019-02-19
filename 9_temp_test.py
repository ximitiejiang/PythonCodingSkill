#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:48:13 2019

@author: ubuntu
"""

def gen_base_anchors(anchor_base, anchor_ratios, anchor_scales):
    """生成n个base_anchors: [xmin,ymin,xmax,ymax]
        xmin = x_center - 
        ymin = 
        xmax = 
        ymax = 
    """
    ratios = torch.tensor(anchor_ratios)
    scales = torch.tensor(anchor_scales)
    
    w = anchor_base
    h = anchor_base
    x_ctr = w*0.5
    y_ctr = h*0.5
    h_ratios = torch.sqrt(ratios)
    w_ratios = 1/h_ratios
    
    ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)

    base_anchors = torch.stack(
    [
        x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
        x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
    ],
    dim=-1).round()
    
    return base_anchors


def gen_base_anchors_mine(anchor_base, ratios, scales, scale_major=True):
    """生成n个base anchors/[xmin,ymin,xmax,ymax],生成的base anchors的个数取决于输入
    的scales/ratios的个数，早期一般输入3个scale和3个ratio,则每个网格包含9个base anchors
    现在一些算法为了减少计算量往往只输入一个scale=8, 而ratios输入3个[0.5, 1.0, 2.0]，
    所以对每个网格就包含3个base anchors
    Args:
        anchor_base(float): 表示anchor的基础尺寸
        ratios(list(float)): 表示h/w，由于r=h/w, 所以可令h'=sqrt(r), w'=1/sqrt(r), h/w就可以等于r了
        scales(list(float)): 表示整体缩放倍数
        scale_major(bool): 表示是否以scale作为anchor变化主体，如果是则先乘scale再乘ratio
    Returns:
        base_anchors(tensor): (m,4)
    1. 计算h, w
        h = base * scale * sqrt(ratio)
        w = base * scale * sqrt(1/ratio)
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


def _meshgrid(x, y, row_major=True):
    xx = x.repeat(len(y))
    yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx
    
def grid_anchors(featmap_size, stride, base_anchors, device='cpu'):
    base_anchors = base_anchors.to(device)
#
    feat_h, feat_w = featmap_size
    shift_x = torch.arange(0, feat_w, device=device) * stride  # 256
    shift_y = torch.arange(0, feat_h, device=device) * stride  # 152
    shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
    shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
    shifts = shifts.type_as(base_anchors)
    # first feat_w elements correspond to the first row of shifts
    # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
    # shifted anchors (K, A, 4), reshape to (K*A, 4)

    all_anchors = base_anchors[None, :, :] + shifts[:, None, :]  # (1,9,4) + (38912,1,4) = (38912, 9, 4)
    all_anchors = all_anchors.view(-1, 4)
    # first A rows correspond to A anchors of (0, 0) in feature map,
    # then (0, 1), (0, 2), ...
    return all_anchors


def grid_anchors_mine(featmap_size, stride, base_anchors):
    """基于base anchors把特征图每个网格所对应的原图感受野都放置base anchors
    Args:
        featmap_size(list(float)): (a,b)
        stride(float): 代表该特征图相对于原图的下采样比例，也就代表每个网格的感受野
                      是多少尺寸的原图网格，比如1个就相当与stride x stride大小的一片原图
        device(str)
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


def valid_flags(featmap_size, valid_size, num_base_anchors, device='cpu'):
    """对all anchors的每个anchor定义合法标志：由于最后一个特征层尺寸是ceil()方式缩小
    该特征层再乘以stride放大回去会比原始图像偏大，这是定义的valid flag就是通过meshgrid
    思想对每个特征图上的网格点生成一个flag，用来标记该网格是否是合法的，如果合法就是1
    不合法就是0。合法代表该网格在原图有对应感受野。(在rpn里边似乎所有的都合法)
    同时由于是对每个网格对应的anchor做，所以还要扩展乘以base-anchor的个数。
    扩展时最关键搞清展平时的过程要跟anchor的排列顺序一致：base_anchors展平时是每层的n个base_anchor先展平
    然后逐层展平，也就是每组base anchor是放在一起的。所以这里的valid复制3组就相当于
    n组base anchor，展平后也是每组anchor的flag放在一起，这个顺序是不能乱的。这也是为什么
    是valid[:,None]而不是valid[None, :], 要放成n列确保n个base anchor展平时是放在一起的。
    Args:
        featmap_size(list): [h,w] 代表网络计算出来的特征图大小，前几层都能被32整除，缩小放大回去尺寸跟原图一样
                            只有最后一层放大后会比原图大
        valid_size(list): [h,w] 代表合法尺寸，该尺寸是通过pad_shape按比例缩小stride倍数
                           后取整ceil()，并与特征图的h,w之间，选择更小的值作为合法尺寸w,h
    Return:
        valid(tensor): (k,) 代表每个网格，但复制了base anchor组并拼接在一行，k数量=网格数量×base anchor数量，保证每个anchor有一个flag
    """
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    v_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
    v_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
    v_x[:valid_w] = 1
    v_y[:valid_h] = 1
    
    v_xx = v_x[:,None].repeat((len(v_y)),1)
    v_yy = v_y[None,:].repeat((1,len(v_x)))
    
    v_xx = v_xx.flatten()  # (k,)
    v_yy = v_yy.flatten()  # (k,)
    
    valid = v_xx & v_yy    # (k,) 只有横坐标纵坐标都为1才为valid
    valid = valid[:,None].repeat(1, num_base_anchors).view(-1)  # 先变换成(k,1)，再复制成(k,3),再展平成(3*k,)的一维数组
    return valid

def inside_flags(anchors, valid_flags, img_shape, allowed_border):
    """对all anchors的边界进行评估，如果超出图像边界，则不使用即标记为0
    Args:
        anchors(tensor): (m,4) 代表特征图上所有anchors
        valid_flags(tensor): (k,) 代表特征图上每个
        img_shape(tuple): [h,w]
        allowed_border(int): 代表anchor超出图像边界的距离，为>=0的整数
    Return:
        inside(tensor): (k,)
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside = valid_flags & \
            (anchors[:, 0] >= -allowed_border) & \
            (anchors[:, 1] >= -allowed_border) & \
            (anchors[:, 2] < img_w + allowed_border) & \
            (anchors[:, 3] < img_h + allowed_border)
    else:
        inside = valid_flags
    return inside


def bbox_overlap(bboxes1,bboxes2,mode='iou'):

    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # (m, 1, 2) vs (n, 2) -> (m,n,2) 代表xmin,ymin
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # (m, 1, 2) vs (n, 2) -> (m,n,2) 代表xmax,ymax

    wh = (rb - lt + 1).clamp(min=0)                       # (m,n,2) - (m,n,2) = (m,n,2) 代表w,h
    overlap = wh[:, :, 0] * wh[:, :, 1]                   # (m,n) * (m,n)
    if mode == 'iou':
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
        ious = overlap / (area1[:, None] + area2 - overlap)
    return ious

def bbox_overlap_mine(bb1, bb2, mode='iou'):
    """bbox的重叠iou计算：iou = Intersection-over-Union交并比(假定bb1为gt_bboxes)
       还有一个iof = intersection over foreground就是交集跟gt_bbox的比
    Args:
        bb1(tensor): (m, 4) [xmin,ymin,xmax,ymax], 通常取bb1输入gt
        bb2(tensor): (n, 4) [xmin,ymin,xmax,ymax], 通常取bb2输入all anchors
    Return:
        ious(tensor): (m,n) 代表的就是m行gt跟n列anchors的ious网格
    1. 计算两个bbox面积：area = (xmax - xmin)*(ymax - ymin)
    2. 计算两个bbox交集：
        >关键是找到交集方框的xmin,ymin,xmax,ymax
        >交集方框的xmin,ymin就等于两个bbox的xmin,ymin取最大值，
         因为这个方框在两个bbox的中间，其取值也就是取大值(画一下不同类型的bbox就清楚了)
         交集方框的xmax,ymax就等于两个bbox的xmax,ymax取最小值，理由同上
        >需要注意要求每个gt_bbox(bb1)跟每个anchor(bb2)的ious, 从数据结构设计上是
         gt(m,4), anchor(n,4)去找max,min, 先取m的一行(m,1,4)与(n,4)比较，
         最后得到(m,n,4)就代表了m层gt,每层都是(n,4),为一个gt跟n个anchor的比较结果。
        >计算交集方框的面积，就是overlap
    3. 计算ious: ious = overlap/(area1+area2-overlap)
    4. 为了搞清楚整个高维矩阵的运算过程中升维/降维的过程，关键在于：
        >抓住[变量组数]m,n的含义，这是不会变的，m和n有时候是层/有时候是行/有时候是列，
         但不会变的是m肯定是gt组数和n肯定是anchor组数
        >抓住运算的目的：如果是按位计算，简单的算就可以了；
         但如果是m组数要轮流跟n组数做运算，那就肯定先升维+广播做一轮运算
        >抓住每一轮输出变量的维度，确保了解这个维度的含义(基于变量组数不变来理解)
    """
    area1 = (bb1[:,2] - bb1[:,0]) * (bb1[:,3] - bb1[:,1]) # (m,)
    area2 = (bb2[:,2] - bb2[:,0]) * (bb2[:,3] - bb2[:,1]) # (n,)
    
    xymin = torch.max(bb1[:, None, :2], bb2[:,:2])  # 由于m个gt要跟n个anchor分别比较，所以需要升维度
    xymax = torch.min(bb1[:, None, 2:], bb2[:,2:])  # 所以(m,1,2) vs (n,2) -> (m,n,2)
    wh = (xymax -xymin).clamp(min=0)   # 得到宽高w, h (m,n,2)
    
    overlap = wh[:,:,0] * wh[:,:,1]   # (m,n)*(m,n) -> (m,n),其中m个gt的n列w, 乘以m个gt的n列h
    
    ious = overlap / (area1[:, None] + area2 -overlap) # 由于m个gt的每一个面积都要跟n的每一个面积相加，要得到(m,n)的面积之和
                                                       # 所以需要升维(m,1)+(n)->(m,n), 然后(m,n)-(m,n)，以及(m,n)/(m,n)都可以操作
    return ious

def assigner(bboxes, gt_bboxes):
    """anchor指定器：用于区分anchor的身份是正样本还是负样本还是无关样本
    正样本标记为1+n(n为index标记), 负样本标记为0, 无关样本标记为-1
    Args:
        bboxes(tensor): (m,4)
        gt_bboxes(tensor): (n,4)
    Return:
        assigned(tensor): (m,) 代表m个bboxes的身份tensor，其值value=[-1,1,2..n]分别表示所对应的gt(-1表示无关，1~n表示第1~n个gt，没有0)
    1. 先创建空矩阵，值设为-1
    2. 再把所有0<iou<0.3的都筛为负样本(0)，iou>0.7的都筛为正样本(1+idx)
    3. 再把该gt最适配的anchor也标为正样本(1+idx)：即gt对应的iou最大的anchor
       注意基于gt找到的iou最高的anchor，往往不是该anchor的最高iou，所以这一步是把anchor中只要高于该iou的所有anchor都提取为fg
    
    """
    pos_iou_thr = 0.7  # 正样本阀值：iou > 0.7 就为正样本
    neg_iou_thr = 0.3  # 负样本阀值：iou < 0.3 就为负样本
    min_pos_iou = 0.3  # 预测值最小iou阀值
    overlaps = bbox_overlap_mine(gt_bboxes, bboxes) # (m,n)代表m个gt, n个anchors
    n_gt, n_bbox = overlaps.shape
    # 第一步：先创建一个与所有anchor对应的矩阵，取值-1(代表没有用的anchor)
    assigned = overlaps.new_full((overlaps.size(1),), -1, dtype=torch.int64)  # (n,)对应n个anchors, 填充-1表示无关样本
                                                                              # 注意这里dtype要改一下，否则跟下面相加的int64冲突
    max_overlap, argmax_overlap = overlaps.max(dim=0)      # (n,)对应n个anchors，表示每个anchor跟哪一个gt的iou最大 (该变量跟assigned同尺寸，用来给assigned做筛选)
    gt_max_overlap, gt_argmax_overlap = overlaps.max(dim=1)# (m,)对应m个gt，表示每个gt跟那个anchor的iou最大
    # 第二步：标记负样本，阀值定义要经可能让负样本数量跟正样本数量相当，避免样本不平衡问题
    assigned[(max_overlap >= 0) & (max_overlap < neg_iou_thr)] = 0  # 0< iou <0.3, value=0
    # 第三步：标记正样本，阀值定义要经可能让负样本数量跟正样本数量相当，避免样本不平衡问题
    # 注意：value = 1 + n, 其中n为第n个gt的意思，所以value范围[1, n_gt+1], value值正好反映了所对应的gt
    assigned[max_overlap >= pos_iou_thr] = 1 + argmax_overlap[max_overlap >= pos_iou_thr] # iou >0.7, value = 1 + 位置值
    # 第四步：标记预测值foreground(也称前景)，也就是每个gt所对应的最大iou为阀值，大于该阀值都算fg
    # 注意：只要取值等于该gt的最大iou都被提取，通常不止一个最大iou。value值范围[1,n_gt+1]代表所对应gt
    for i in range(n_gt):
        if gt_max_overlap[i] >= min_pos_iou:
            max_iou_idx = overlaps[i]==gt_max_overlap[i] # 从第i行提取iou最大的位置的bool list
            assigned[max_iou_idx] = 1 + i   # fg的value比正样本的value偏小
    return assigned
    


def random_sampler(assigned, bboxes):
    """anchor抽样器: 基于随机采样方式,从all anchors里边先分离出正样本和负样本，
    然后在正负样本中分别按照比例抽取总数固定的样本个数用于训练(通常抽需256个样本)
    Args:
        assigned(tensor): (m,) 代表m个anchor的身份指定, 取值范围[-1,1,2,..n]
        bboxes()
    Return:
        pos_inds(tensor): (j,) 代表指定数量的anchor正样本的index列表
        neg_inds(tensor): (k,) 代表指定数量的anchor负样本的index列表
        
    """
    num_expected = 256   # 总的采样个数
    pos_fraction = 0.5   # 正样本占比
    num_expect_pos = int(num_expected * pos_fraction)
    # 正样本抽样：通常正样本数量较少，不会超过num ecpected
    pos_inds = torch.nonzero(assigned > 0)  # (j,1) 正样本的index号
    if torch.numel(pos_inds)!=0:
        pos_inds = pos_inds.squeeze(1)      # (j,)
    if torch.numel(pos_inds) > num_expect_pos:  # 如果正样本数太多则抽样
        pos_rand_inds = torch.randperm(len(pos_inds))[:num_expect_pos] # 先对index随机排序，然后抽取前n个数
        pos_inds = pos_inds[pos_rand_inds]
        num_sampled_pos = num_expect_pos
    else:   # 如果正样本数太少，则更新实际采样的数量num_sampled_pos，从而负样本数会增加，保证总数固定
        num_sampled_pos = len(pos_inds)
#        candidates = np.arrange(len(pos_inds))  # 也可用numpy来实现采样，速度比torch快
#        np.random.shuffle(candidates)
#        rand_inds = cnadidates[:num_expected*pos_fraction]
#        return pos_inds[]
    # 负样本抽样：通常负样本数量较多，所有anchors里边可能70%以上iou都>0，即都为负样本
    neg_inds = torch.nonzero(assigned == 0) # (k, 1)负样本的index号
    if torch.numel(neg_inds)!=0:
        neg_inds = neg_inds.squeeze(1)      # (k,)
    if torch.numel(neg_inds) > num_expected - num_sampled_pos:
        neg_rand_inds = torch.randperm(len(neg_inds))[:(num_expected - num_sampled_pos)]
        neg_inds = neg_inds[neg_rand_inds]
    return pos_inds, neg_inds

    
def bbox2delta(prop, gt, mean=[0,0,0,0], std=[1,1,1,1]):
    """对每个gt bbox所对应的多个proposal anchor进行回归：关键理解是多个anchor对应了1个gt bbox
    那么这些proposal是分布在gt的周围，需要找到一个回归模型的参数，使用回归后的anchor坐标作为预测值
    这就好比：一组散点(多个proposal anchor)先拟合出一条回归曲线，然后用回归曲线计算出的值作为预测值
    在做proposal anchor的回归前，先把xmin/ymin/xmax/ymax转化为x/y/w/h的原因，是因为???
    Args:
        prop(tensor): (j,4) 代表正样本anchors的坐标(xmin,ymin,xmax,ymax)
        gt(tensor): (j,4) 代表每一个正样本anchors所预测对应的gt的坐标
        mean(list)
        std(list)
    Returns:
        deltas(tensor): (j,4) 代表了从proposal到gt的回归参数(因为两者很接近，回归参数也很小，一般都是小于1的小数)
    """
    # 把proposal的bbox坐标xmin,ymin,xmax,ymax转换xctr,yctr,w,h
    px = 0.5 * (prop[...,0] + prop[...,2])  # (j,)
    py = 0.5 * (prop[...,1] + prop[...,3])
    pw = prop[...,2] - prop[...,0]
    ph = prop[...,3] - prop[...,1]
    # 把gt的bbox坐标xmin,ymin,xmax,ymax转换xctr,yctr,w,h
    gx = 0.5 * (gt[...,0] + gt[...,2])   
    gy = 0.5 * (gt[...,1] + gt[...,3])
    gw = gt[...,2] - gt[...,0]
    gh = gt[...,3] - gt[...,1]
    # 计算二者的
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx,dy,dw,dh], dim=-1)  # (j, 4)
    
    mean = deltas.new_tensor(mean).unsqueeze(0)
    std = deltas.new_tensor(std).unsqueeze(0)
    deltas = deltas.sub_(mean).div_(std)
    
    return deltas

def unmap(d1,d2,d3):
    pass

def anchor_target_mine(gt_bboxes, anchors, assigned, pos_inds, neg_inds, inside_f, gt_labels):
    """anchor目标：首先对anchor的合法性进行过滤，取出合法anchors(没有超边界)，
    注意，这里的anchors需要是valid anchors，同时传入assigner/sampler的也应该是valid anchors
    Args:
        gt_bboxes(tensor): (m,4) 代表标签bboxes
        anchors(tensor): (n,4) 代表所有网格上的anchors, 每个网格上有9个base anchors
        assigned(tensor): (n,) 指定器输出结果，代表n个anchor的身份指定[-1,0,1,2..m]
        pos_inds(tensor): (j,) 采样器输出结果，代表j个采样得到的正样本anchors的index
        neg_inds(tensor): (k,) 采样器输出结果，代表k个采样得到的负样本anchors的index
        inside_f(tensor): (n,) 对anchor在图像边界内部的判断结果[0,1]，每个anchor一个flag
        
    Return:
        labels
        labels_weights
        bbox_targets
        bbox_weights
    """
    # 先基于inside_flag获得inside anchors: 代表的是在图像边界以内的anchors
    inside_anchors = anchors[inside_f,:]
    # 先创建0数组
    bbox_targets = torch.zeros_like(inside_anchors)  # (n,4)
    bbox_weights = torch.zeros_like(inside_anchors)  # (n,4)
    labels = anchors.new_zeros(inside_anchors.shape[0],dtype=torch.int64) # (n,)
    labels_weights = anchors.new_zeros(inside_anchors.shape[0], dtype= torch.float32) # (n,)
    # 采样index转换为bbox坐标
    pos_bboxes = anchors[pos_inds]  # (j,4)正样本index转换为bbox坐标
    # 生成每个正样本所对应的gt坐标，用来做bbox回归
    pos_assigned = assigned[pos_inds] - 1       # 提取每个正样本所对应的gt(由于gt是大于1的1,2..)，值减1正好就是从0开始第i个gt的含义
    pos_gt_bboxes = gt_bboxes[pos_assigned,:]   # (j,4) 生成每个正样本所对应gt的坐标
    if len(pos_inds) > 0:
        #对正样本相对于gt做bbox回归
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes) # (j, 4)得到的是每个proposal anchor对应回归target的回归参数
        # 更新bbox_targets/bbox_weights
        bbox_targets[pos_inds, :] = pos_bbox_targets  # 所有anchor中正样本的坐标更新为targets的deltas坐标
        bbox_weights[pos_inds, :] = 1.0               # 所有anchor中正样本的权重更新为1
        # 更新labels/labels_weights
        labels[pos_inds] = 1            # 默认gt_labels=None，所以labels对应target的位置设置为1
        labels_weights[pos_inds] = 1.0  # cfg中pos_weight可自定义，如果定义-1说明用默认值则设为1
    if len(neg_inds) > 0:
        labels_weights[neg_inds] = 1.0

    # unmap: 采用默认的unmap_outputs =True
    num_total_anchors = anchors.size(0)
    labels = unmap(labels, num_total_anchors, inside_flags)
    labels_weights = unmap(labels_weights, num_total_anchors, inside_flags)
    
    return labels, labels_weights, bbox_targets, bbox_weights
    
    

#-------------base anchors---------------------   
import torch    
anchor_strides = [4., 8., 16., 32., 64.]
anchor_base_sizes = anchor_strides      # 基础尺寸
#anchor_scales = [8., 16., 32.]          # 缩放比例
anchor_scales = [8.]                    # 只传入1个scale，减少anchors的个数
anchor_ratios = [0.5, 1.0, 2.0]         # w/h比例

num_anchors = len(anchor_scales) * len(anchor_ratios)
base_anchors = []
base_anchors2 = []
for anchor_base in anchor_base_sizes:
    base_anchors.append(gen_base_anchors_mine(anchor_base, anchor_ratios, anchor_scales))
    base_anchors2.append(gen_base_anchors(anchor_base, anchor_ratios, anchor_scales))
    
#-------------all anchors---------------------
featmap_sizes = [(152,200), (76,100), (38,50), (19,25), (10,13)]
strides = [4,8,16,32,64]    # 针对resnet的下采样比例，5路分别缩减尺寸 

i=0
featmap_size = featmap_sizes[i]
stride = strides[i]
base_anchor = base_anchors[i]
all_anchors1 = grid_anchors(featmap_size, stride, base_anchor)
all_anchors2 = grid_anchors_mine(featmap_size, stride, base_anchor)

#-------------flags---------------------
valid_size = featmap_size   # 这里沿用rpn的形式，由于加入对数据32倍数的padding,特征图大小跟valid size一样
num_base_anchors = len(base_anchor)
allowed_border = 0
img_shape = (600, 800)

valid_f = valid_flags(featmap_size, valid_size, num_base_anchors, device='cpu')

inside_f = inside_flags(all_anchors2, valid_f, img_shape, allowed_border)

#-------------bbox ious---------------------
#bb1 = torch.tensor([[-20.,-20.,20.,20.],[-30.,-30.,30.,30.]])
#bb2 = torch.tensor([[-25.,-25.,25.,25.],[-15.,-15.,15.,15.],[-25,-25,50,50]])
#ious1 = bbox_overlap(bb1,bb2)
#ious2 = bbox_overlap_mine(bb1, bb2)

#-------------ious---------------------
sampling =True  # 调用anchor target时没指定就沿用默认设置
import pickle
gt_bboxes = pickle.load(open('test/test_data/test9_bboxes.txt','rb'))  # gt bbox (m,4)
gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)
gt_labels = [None, None]
ious = bbox_overlap_mine(gt_bboxes, all_anchors2)

#-------------ious/assigner/sampler---------------------
assigned = assigner(all_anchors2, gt_bboxes)

pos_inds, neg_inds = random_sampler(assigned, all_anchors2)

#-------------anchor target---------------------
labels, labels_weights, bbox_targets, bbox_weights = anchor_target_mine(
        gt_bboxes, all_anchors2, assigned, pos_inds, neg_inds, inside_f, gt_labels)



"""一组参考数据(2张图片)
# 基于pad_shape进行下采样缩放得到的5张特征图，由于前面padding到32倍数，所以下采样不需要取整
# 但最后一张特征图是64倍，所以相除结果必然是.5，采用ceil的方式得到(10,13)
# 我理解为什么没有用64倍做padding，是防止太多padding的0进去
featmap_sizes = [[152, 200], [76, 100], [38, 50], [19, 25], [10, 13]]

ori_shape = (375, 500, 3)  # 原图(h,w,c)
img_shape = (600, 800, 3)  # 图片放大到[1000,600]以内
pad_shape = (608, 800, 3)  # 图片加padding能够被32整除(但rpn最大下采样比例是64?)
scale_factor = 1.6
flip = False

ori_shape = 375, 500, 3
img_shape = 600, 800, 3
pad_shape = 608, 800, 3
scale_factor = 1.6
flip = True        
"""

