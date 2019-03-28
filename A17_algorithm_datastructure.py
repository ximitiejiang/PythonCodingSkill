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
        base_size(int): 基础方框的尺寸
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
def grid_anchors():
    """网格化anchors到featmap的每一个cell上面，即生成所有anchors()
    Args:
        featmap_size
        stride()
        base_anchors(tensor)
    """
    

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
"""如何进行bbox回归?即如何做bbox2delt，以及如何回复bbox坐标?
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
    run_focal_loss = True
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
    areas = 
    

def soft_nms():
    pass



