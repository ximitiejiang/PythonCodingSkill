#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:01:02 2019

@author: ubuntu
"""

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
    Args:
        featmap_size
        stride()
        base_anchors(tensor): (m,4)(xmin,ymin,xmax,ymax)
    """
    x = torch.arange(0, featmap_size[0], stride)
    y = torch.arange(0, featmap_size[1], stride).reshape(-1,1)
    xx = x.repeat(len(y), dim = 0).reshape(-1)
    yy = y.repeat(len(x), dim = 1).reshape(-1)
    
    anchors = [base_anchors[:,0] + xx,
               base_anchors[:,2] + yy,
               base_anchors[:,1] + xx,
               base_anchors[:,3] + yy]
    return anchors

if __name__ == '__main__':
    run_grid_anchor = True
    if run_grid_anchor:
        featmap_size = [320,320]
        stride = 8
        base_anchors = [[-10,-10,10,10],[-20,-20,20,20]]
        grid_anchors(featmap_size, stride, base_anchors)

# %% 核心算法3
"""如何计算iou支持assigner?
"""
def ious(bb1, bb2):
    """把bb1, bb2逐个进行iou计算
    Args:
        bb1(tensor)
        bb2(tensor)
    Returns:
        iou
    """


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
def focal_loss(preds, targets, alpha=0.25, gamma=2, avg_factor):
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
"""
def nms():
    """对输入的proposal进行nms过滤
    Args:
        proposals(tensor): (m,5)代表(xmin,ymin,xmax,ymax, score)
        iou_thr(float): 代表iou重叠的阀值，超过该值就认为是重叠，通常取0.7
    Returns:
        keep(tensor): ()
    """
    xmin
    ymin
    xmax
    ymax
    areas
    

def soft_nms():
    pass


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
