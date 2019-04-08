#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:01:02 2019

@author: ubuntu
"""
# %% 训练框架
"""如何写一个训练框架？
5步走：
1. 配置
2. 模型
3. 数据
4. 训练设置
5. 训练: 主要就是在两层循环里边调整optimizer的相关设置，并做loss反向传播
    2层循环，
    set_optimizer_lr()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""
import Config
import torch
import torch.nn as nn
import CocoDataset, VOCDataset
import OneStageDetector
import GroupSampler

'基于cfg的type和params创建对象'
def get_dataset(cfg_data):
    """返回一个数据集对象: 保留repeat dataset功能，保留多数据集组合功能
    """
    if cfg_data.pop('type') == 'CocoDataset':
        dataset = CocoDataset(cfg_data)
    elif cfg_data.pop('type') == 'VOCDataset':
        dataset = VOCDataset(cfg_data)  
    return dataset
    
def get_optimizer(cfg_opt):
    """返回优化器对象：
    """
    obj_type = cfg_opt.pop('type')                       # 获得类型字符串
    obj_type = getattr(torch.optim.Optimizer, obj_type)  # 从父类获得类：但Optimizer跟比如SGD类应该还是有区别吧，能代替子类来创建optimizer吗？
    return obj_type(cfg_opt)
    
def set_lr(optimizer, cfg_lr):
    """设置优化器学习率
    """
    lr_groups = []
    for param_group, lr in zip(optimizer.param_groups, lr_groups):  # optimizer.param_groups就是参数dict
        param_group['lr'] = lr

def batch_processor(model, data_batch):
    """进行每个batch data的核心计算,主要是求解loss,计算loss求和
    """
    losses = model(**data_batch)
    loss = losses.sum()
    outputs = dict(loss = loss)
    return outputs

def collat_func():
    """进行数据的堆叠
    """
    pass

def train(cfg_path, dataset_class):
    '1. cfg'
    cfg = Config.fromfile(cfg_path)
    torch.backends.cudnn.benchmark = True   # 目的是让内置cudnn自动寻找最合适的高效算法来优化运行效率，如果网络输入数据的维度/类型变化不大，可设置为True,但如果输入数据在每个iter维度/类型变化，则导致cudnn每次要去寻找一遍最优配置，反而会降低运行效率
    '2. model'
    model = OneStageDetector(cfg)
    if cfg.parallel:
        model = nn.DataParallel(model)
    '3. data'
    batch_size = 2
    num_workers = 2
    dataset = get_dataset(cfg.data.train, cfg.data.test)    
    dataloader = nn.DataLoader(dataset, 
                            batch_size = batch_size,
                            sampler = GroupSampler(),
                            num_workers=num_workers,
                            collate_fn=collat_func,
                            pin_memory=False)
    '4. training set'
    optimizer = get_optimizer(cfg.optimizer)  # 训练设置都封装在runner.init()
    epoch=0
    iter=0
    
    while epoch < cfg.epoches:   # epoch部分封装在runner.run()    
        '5. train'
        model.train()            # 其他封装在runner.train()
        '5.1 before epoch'
        set_lr()
        
        for i, data_batch in enumerate(dataloader):
            '5.2 before iter'
            outputs = batch_processor(model, data_batch)
            
            '5.3 after iter'   # 可通过hook实现
            optimizer.zero_grad()        # 梯度清零
            outputs['loss'].backward()   # 更新梯度
            optimizer.step()             # 更新参数
            
            iter += 1
        '5.4 after epoch'
        epoch += 1
    
if __name__ == '__main__':
    run_train = True
    if run_train:
        cfg_path = './config/cfg_ssd300_voc.py'
        train(cfg_path, CocoDataset)


# %% 核心算法1
"""如何生产base anchors?
""" 
import numpy as np 
import torch
def gen_base_anchors(base_size, ratios, scales):
    """生成一组base anchors(m,4): 基于一组ratios/scales进行排列组合
    注意：通常按照ratio优先的原则，也就是1个scale先对应一轮ratios，然后第二个scale对应一轮ratios
    因为很多算法里边往往只需要一个scale对应的一轮ratios，第二轮则只取一个scale, 这样写便于截取
    所以是每一个scale对应j列ratios,也就是k行scale对应j列ratio, 也就是生成(k,j)
    Args:
        base_size(int): 基础方框的尺寸, 通常区stride的值
        ratios(array): (j,)代表h/w的比值, 涉及numpy的广播机制，所以必须是array而不是list
        scales(array): (k,)代表相对于base_size的缩放比例, 涉及numpy的广播机制，所以必须是array而不是list
    Returns:
        base_anchors(tensor): (k*j, 4)
    """
    h = base_size
    w = base_size
    x_ctr = 0.5 * w
    y_ctr = 0.5 * h
    h_ratios = np.sqrt(ratios)   
    w_ratios = 1 / h_ratios

    h_new = h * scales[:, None] * h_ratios[None, :]   # (k,)*(j,)->(k,1)*(1,j)->(k,j)*(k,j)->(k,j)
    w_new = w * scales[:, None] * w_ratios[None, :]
    base_anchors = np.stack([x_ctr - 0.5 * w_new,
                             y_ctr - 0.5 * h_new,
                             x_ctr + 0.5 * w_new,
                             y_ctr + 0.5 * h_new], axis=0)
    return torch.tensor(base_anchors.astype(np.int32))
    
if __name__=='__main__':
    run_base_anchor = False
    if run_base_anchor:
        base_size = 32
        ratios = np.array([1,1/2,2])
        scales = np.array([1,2,3])
        base_anchors = gen_base_anchors(base_size, ratios, scales)
    
    
# %% 核心算法2
"""如何grid anchors?
"""
def grid_anchors(featmap_size, stride, base_anchors):
    """网格化anchors到featmap的每一个cell上面，即生成所有anchors()
    本质上就是要把base_anchors平移到所有cell上去，也就是先生成所有cell的坐标，然后base_anchor
    中的每一个anchor都要跟所有cell的坐标相加
    Args:
        featmap_size
        stride()
        base_anchors(tensor): (m,4)(xmin,ymin,xmax,ymax)
    """
    x = torch.arange(0, featmap_size[0]) * stride
    y = (torch.arange(0, featmap_size[1]) * stride).reshape(-1,1)
    xx = x.repeat(len(y), 1).reshape(-1)
    yy = y.repeat(1, len(x)).reshape(-1)
    
    delta = torch.stack([xx, yy, xx, yy], dim = -1)  # (n,4) 偏移量
    # (m,4) + (n,1,4) = (m,4) + (n,m,4) = (n,m,4)
    anchors = base_anchors + delta[:,None,:]         # 一组base anchor要同时平移到每一个cell,所以需要广播
    anchors = anchors.reshape(-1,4)          
    
    return anchors

if __name__ == '__main__':
    run_grid_anchor = False
    if run_grid_anchor:
        featmap_size = [320,320]
        stride = 8
        base_anchors = torch.tensor([[-10,-10,10,10],[-20,-20,20,20]])
        grid_anchors(featmap_size, stride, base_anchors)

# %% 核心算法3
"""如何计算iou支持assigner?
"""
def ious(bb1, bb2):
    """把bb1, bb2逐个进行iou计算，关键要搞清广播原则计算
    Args:
        bb1(tensor) (m,4)
        bb2(tensor) (n,4)
    Returns:
        iou(tensor) (m,n)
    """
    area1 = (bb1[:,3] - bb1[:,1]) * (bb1[:,2] - bb1[:,0]) # (m,)
    area2 = (bb2[:,3] - bb2[:,1]) * (bb2[:,2] - bb2[:,0]) # (n,)
    # m个bb1,n个bb2,所以ious必然是(m,n)个，所以从求解xymin开始就要得到(m,n)个
    xymin = torch.max(bb1[:, None, :2], bb2[:,:2])  # (m,2) maxwith (n,2) -> (m,1,2)vs(n,2) -> (m,n,2)
    xymax = torch.min(bb1[:, None, 2:], bb2[:,2:])  # (m,n,2)
    
    w = xymax[:,:,0] - xymin[:,:,0] # (m,n)
    h = xymax[:,:,1] - xymin[:,:,1] # (m,n)
    overlap = w * h  # (m,n)
    
    cal_iou = overlap / (area1[:,None] + area2 + overlap)  # (m,n)/(m,1)+(n,)-(m,n)
    
    return cal_iou
    
if __name__ == '__main__':
    run_ious = False
    if run_ious:
        bb1 = torch.tensor([[-20.,-20.,20.,20.],[-18.,-18.,18.,18.]])
        bb2 = torch.tensor([[-19.,-19.,23.,23.],[-19.,-19.,22.,22.]])
        iou = ious(bb1, bb2)


# %% 核心算法4
"""如何进行bbox回归?即如何得到bbox target，以及如何回复bbox坐标?
"""
def bbox2delta():
    pass

# %% 核心算法5
"""如何把labels转换成binary的独热编码(即概率形式)?
"""
def bin_labels(labels, num_classes):
    labels_new = torch.zeros(len(labels), num_classes)
    for i, label in enumerate(labels):
        ind = label.numpy().item()
        labels_new[i][ind] = 1
    return labels_new

if __name__=='__main__':
    run_bin_labels = False
    if run_bin_labels:
        labels = torch.tensor([1,0,7])
        num_classes = 80
        new_labels = bin_labels(labels, num_classes)

# %% 核心算法6
"""如何写交叉熵损失函数和focal loss?
"""
import torch.nn.functional as F
def focal_loss(preds, targets, alpha=0.25, gamma=2, avg_factor=1):
    """输入多分类概率形式的preds和概率形式的targets
    公式focal loss = - alpha_t * (1 - pt)^gamma * log(pt) 
    Args:
        preds(m, 80)
        targets(m, 80)
        weights(m, 80)
    Returns:
        loss(tensor float)
    """
    pt = preds * targets + (1 - preds) * (1 - targets)
    alpha_t = alpha * targets + (1- alpha) * (1 - targets)
    weights = alpha_t * (1 - pt).pow(gamma)
    loss = weights * F.binary_cross_entropy_with_logits(preds, targets)
    return loss/avg_factor
    
if __name__=='__main__':
    run_focal_loss = False
    if run_focal_loss:
        preds = torch.randn(3,80)        # (m,80)
        targets = torch.tensor([1,0,7])
        avg_factor = 3
        targets_new = bin_labels(targets, num_classes=80)  # (m,80)
        
        loss = focal_loss(preds, targets_new, avg_factor)
        
        

# %% 核心算法7
"""如何进行非极大值抑制nms以及如何做soft_nms?
1. 对于普通nms: 基于score排序的index, 保存第一个，与剩余的做ious, 
"""
def nms(proposals, iou_thr):
    """对输入的proposal进行nms过滤
    Args:
        proposals(tensor): (m,5)代表(xmin,ymin,xmax,ymax, score),该score是通过sigmoid计算得到的概率
        iou_thr(float): 代表iou重叠的阀值，超过该值就认为是重叠，通常取0.7
    Returns:
        keep(tensor): ()
    """
    xmin = proposals[:,0]
    ymin = proposals[:,1]
    xmax = proposals[:,2]
    ymax = proposals[:,3]
    areas = (ymax-ymin+1) * (xmax-xmin+1)
    scores = proposals[:,4]
    keep = []
    index = scores.argsort()[::-1]  # 先从大到小排序，返回index
    while index.size >0:
#        i = index[0]        
        keep.append(index[0])     # 每轮循环提取score最高的第一个值作为对象，并保存index   
        x11 = np.maximum(xmin[index[0]], xmin[index[1:]])    # 计算保存的这个bbox与其他所有bbox的iou
        y11 = np.maximum(ymin[index[0]], ymin[index[1:]])    # (1,4)vs(n,4)
        x22 = np.minimum(xmax[index[0]], xmax[index[1:]])
        y22 = np.minimum(ymax[index[0]], ymax[index[1:]])        
        w = np.maximum(0, x22-x11+1)    # the weights of overlap
        h = np.maximum(0, y22-y11+1)    # the height of overlap       
        overlaps = w * h        
        ious = overlaps / (areas[index[0]]+areas[index[1:]] - overlaps)        
        idx = np.where(ious<=iou_thr)[0]   # 查找所有ious小于阀值的index保留下来，其他大于阀值的index就相当于丢掉了        
        index = index[idx+1]   # because index start from 1       
    return keep

def nms_tensor(proposals, iou_thr):
    """
    """
    area2 = (proposals[index[0],2] -proposals[index[0],0]) * \
            (proposals[index[0],2] -proposals[index[0],0])
    scores = proposals[:,4]
    index = torch.argsort(scores)[::-1]
    keep = []
    while index.size() > 0:
        keep.append(index[0])
        
        area1 =  
        xymin = proposals[]

def soft_nms():
    pass

if __name__ == '__main__':
    proposals = torch.tensor([[-20,-20,20,20],
                              [-25,-25,10,10],
                              [-10,-10,30,30],
                              [-23,-23,18,18]])
    nms_tensor(proposals, iou_thr=0.7)


# %% 核心算法
"""如何进行hard negtive mining?

"""    
def hard_negtive_mining(loss_cls_all, pos_inds, neg_inds, num_total_samples):
    """通过hard negtive mining筛选
    1. 所有targets都进行loss计算，其中正样本loss很少(20+)，负样本loss大量存在(8000+)
    2. 正样本loss全部保留(20+)，负样本loss个数取正样本的3倍个数，且只取负样本loss的排名前面的3倍loss(60+)
    3. 缩减loss: 分别求正样本loss之和，负样本loss之和
    4. 平均loss: 平均因子取label中正样本个数(注意坑：label里边正样本就是label>0，
       有可能target的正样本数量少于label正样本数量，所以一定要取label的正样本个数而不是target正样本个数)
    """
    num_pos_samples = pos_inds.size(0)     # 获得正样本个数
    num_neg_samples = num_pos_samples * 3  # 获得负样本个数
    if num_neg_samples > neg_inds.size(0): # 如果负样本数量太少，少于正样本的3倍，则负样本数量只能取实际负样本数量
        num_neg_samples = neg_inds.size(0)
    
    topk_loss_neg = loss_cls_all[neg_inds].topk(num_neg_samples)  # 取负样本损失的前k个
    
    loss_cls_pos = loss_cls_all[pos_inds].sum()   # loss缩减
    loss_cls_neg = topk_loss_neg.sum()            # loss缩减
    loss_cls = (loss_cls_pos + loss_cls_neg)/num_total_samples    # loss平均
