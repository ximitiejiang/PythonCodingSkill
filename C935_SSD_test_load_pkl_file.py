#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:29:43 2019

@author: ubuntu
"""
# %%
"""检测器在整个coco数据集上的评估方法？
1. coco数据集的评估从官网上看需要评估12个参数如下：
参考：https://www.jianshu.com/p/d7a06a720a2b (非常详细介绍了两大数据集检测竞赛的评价方法包括源码)
参考：https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py (coco官网的eval代码)
注意：coco并不区分AP和mAP，以及AR和mAR，一般都是指mAP(均值平均精度)即所有类别的平均精度
并且AP是coco最核心的一个指标，AP高的取胜。
(AP)Average Precision
    AP              # 在iou = [0.5,0.95,0.05]范围内的平均AP
    AP IoU=0.5      # 在iou = 0.5的平均AP(这也是voc的要求)
    AP IoU=0.75     # 在iou = 0.75的平均AP(更严格的要求)
(AP)Average Precision across scales:
    AP small        # 小目标的检测AP(area< 32*32), 约41%
    AP medium       # 中目标的检测AP(32*32<area<96*96), 约34%
    AP large        # 大目标的检测AP(area> 96*96), 约24%
    其中面积通过mask像素数量计算
(AR)Average Recall
    AR max=1        # 一张图片图片给出最多1个预测
    AR max=10       # 一张图片图片给出最多10个预测
    AR max=100      # 一张图片图片给出最多100个预测
(AR)Average Recall across scales:
    AR small        # 小目标的召回率
    AR medium       # 中目标的召回率
    AR large        # 大目标的召回率

2. voc的评价方法虽然也是box AP为主，但计算方法稍有不同：
AP专指IoU=0.5时
"""
from six.moves import cPickle as pickle
import numpy as np
from pycocotools.coco import COCO
from terminaltables import AsciiTable
from B03_dataset_transform import vis_bbox

#import _pickle as pickle
"""注意cPickle, Pickle, six.moves的区别：
1. cPickle是c代码写成，Pickle是python写成，相比之下cPickle更快
2. cPickle只在python2中存在，python3中换成_pickle了
3. six这个包是用来兼容python2/python3的，这应该是six的由来(是2与3的公倍数)
   six包里边集成了有冲突的一些包，所以可以从里边导入cPickle这个在python3已经取消的包
"""

def evaluation(result_file_path):
    """基于已经生成好的pkl模型预测文件，进行相关操作
    1. 跟numpy version是否有关：mac为1.14.2（numpy.version.version)，而ubuntu的numpy是1.16.0
    2. 是否换一种文件格式，不用pkl格式
    """
    pass

def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def _recalls(all_ious, proposal_nums, thrs):

    img_num = all_ious.shape[0]
    total_gt_num = sum([ious.shape[0] for ious in all_ious])

    _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros((ious.shape[0]))
            if ious.size == 0:
                tmp_ious = np.hstack((tmp_ious, gt_ious))
                continue
            for j in range(ious.shape[0]):
                gt_max_overlaps = ious.argmax(axis=1)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious

    _ious = np.fliplr(np.sort(_ious, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num)

    return recalls


def set_recall_param(proposal_nums, iou_thrs):
    """Check proposal_nums and iou_thrs and set correct format.
    """
    if isinstance(proposal_nums, list):
        _proposal_nums = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        _proposal_nums = np.array([proposal_nums])
    else:
        _proposal_nums = proposal_nums

    if iou_thrs is None:
        _iou_thrs = np.array([0.5])
    elif isinstance(iou_thrs, list):
        _iou_thrs = np.array(iou_thrs)
    elif isinstance(iou_thrs, float):
        _iou_thrs = np.array([iou_thrs])
    else:
        _iou_thrs = iou_thrs

    return _proposal_nums, _iou_thrs


def eval_recalls(gts,
                 proposals,
                 proposal_nums=None,
                 iou_thrs=None,
                 print_summary=True):
    """Calculate recalls.

    Args:
        gts(list or ndarray): a list of arrays of shape (n, 4) 总计5000个，代表每张图有一组gt bboxes
        proposals(list or ndarray): a list of arrays of shape (k, 4) or (k, 5)
        proposal_nums(int or list of int or ndarray): top N proposals
        thrs(float or list or ndarray): iou thresholds

    Returns:
        ndarray: recalls of different ious and proposal nums
    """

    img_num = len(gts)
    assert img_num == len(proposals)
    # 这里proposals是5000张图的所有预测bboxes汇总
    # proposal_nums是看提取前top100,top300, top1000 ???
    proposal_nums, iou_thrs = set_recall_param(proposal_nums, iou_thrs)

    all_ious = []
    for i in range(img_num):
        
        if isinstance(proposals[i], list):  # 新增一个分支
            new_prop = []
            for j in range(len(proposals[i])):
                if proposals[i][j].size != 0:
                    new_prop.append(proposals[i][j])
            if len(new_prop) !=0:
                img_proposal = np.concatenate(new_prop, axis=0)
            else:
                img_proposal = np.zeros((0,4),dtype=np.float32)
        
        elif proposals[i].ndim == 2 and proposals[i].shape[1] == 5: # 如果一张图片的proposal是(n,5)形式
            scores = proposals[i][:, 4]               # 取出最后一列scores置信度 
            sort_idx = np.argsort(scores)[::-1]       # scores排序，从大到小
            img_proposal = proposals[i][sort_idx, :]  # 取出按score排序出的proposals
        else:
            img_proposal = proposals[i]
    
        prop_num = min(img_proposal.shape[0], proposal_nums[-1]) # 如果proposal个数少于则取少的那个值
        if gts[i] is None or gts[i].shape[0] == 0:  # 如果没有gt
            ious = np.zeros((0, img_proposal.shape[0]), dtype=np.float32)
        else:
            ious = bbox_overlaps(gts[i], img_proposal[:prop_num, :4]) # 计算gt与proposals的ious
                                                                      # 也就要求proposal[i]需要是一张图片所有class的总和
        all_ious.append(ious)
    all_ious = np.array(all_ious)
    recalls = _recalls(all_ious, proposal_nums, iou_thrs)
    if print_summary:
        print_recall_summary(recalls, proposal_nums, iou_thrs)
    return recalls


def print_recall_summary(recalls,
                         proposal_nums,
                         iou_thrs,
                         row_idxs=None,
                         col_idxs=None):
    """Print recalls in a table.

    Args:
        recalls(ndarray): calculated from `bbox_recalls`
        proposal_nums(ndarray or list): top N proposals
        iou_thrs(ndarray or list): iou thresholds
        row_idxs(ndarray): which rows(proposal nums) to print
        col_idxs(ndarray): which cols(iou thresholds) to print
    """
    proposal_nums = np.array(proposal_nums, dtype=np.int32)
    iou_thrs = np.array(iou_thrs)
    if row_idxs is None:
        row_idxs = np.arange(proposal_nums.size)
    if col_idxs is None:
        col_idxs = np.arange(iou_thrs.size)
    row_header = [''] + iou_thrs[col_idxs].tolist()
    table_data = [row_header]
    for i, num in enumerate(proposal_nums[row_idxs]):
        row = [
            '{:.3f}'.format(val)
            for val in recalls[row_idxs[i], col_idxs].tolist()
        ]
        row.insert(0, num)
        table_data.append(row)
    table = AsciiTable(table_data)
    print(table.table)


if __name__=='__main__':
    eval_with_pkl_file = True
    """假定result.pkl已经获得则可按如下进行评估，但实际的test forward()计算过程如下
    在detector的forward_test()函数中, 内部调用simple_test()
        - 从backbone/neck获得x: 从img(1,3,800, 1216)到x[(1,256,200,304),(1,256,100,152),(1,256,50,76),(1,256,25,38),(1,256,13,39)]
        - 再从RPN head调用simple_test_rpn()
            获得rpn head的输出rpn_outs = rpn_head(x), 输出结构2个元素，[cls_scores, bbox_preds], 每个都是5层
            获得proposal_list = rpn_head.get_bboxes(), 输出结构1个元素，[(2000,5)]
        - 再从Bbox head调用simple_test_bboxes()
            获得rois = bbox2rois(proposals), 输出结构(2000,5)
            获得roi_feats = bbox_roi_extractor(x, rois)，输出结构(2000,256,7,7)
            获得bbox head的输出cls_score/bbox_pred = bbox_head(roi_feats)，输出结构(2000,81)和(2000,324)
            获得det_bboxes, det_labels = bbox_head.get_det_bboxes()，输出结构(100, 5)和(100,)
         - 最后调用bbox2result()从det_bboxes, det_labels中筛选出results, 结构为[class1, class2, ...]，
           每个class为(n,5)数据，代表预测到的该类的bbox个数和置信度
         - 对于整个数据集的single_test，一张图片会对应一个result，所以：
           最终results list长度5000，每个result长度80，每个类读应bbox array(n,5)    
    """
    # 打开pkl获得数据
    if eval_with_pkl_file:
        # 测试一下gt bboxes的情况
        f = open('coco_gt_bboxes.pkl','rb') # [array1, ...array5000]共5000张图的gt bboxes(xyxy格式)
        gtb = pickle.load(f)
        vis_bbox(None, gtb[0])  # 第一个None表示没有图片，但为什么显示不出来那些gt bboxs?
        f.close()
        
        data_root = 'data/coco/'    # 需要预先把主目录加进sys.path
        ann_file=[data_root + 'annotations/instances_train2017.json',
                  data_root + 'annotations/instances_val2017.json']
        img_prefix=[data_root + 'train2017/', data_root + 'val2017/']
        
        eval_types = ['proposal_fast']
        result_file_path = 'data/coco/results.pkl'
#        result_file_path = 'data/VOCdevkit_mac/results.pkl'
        
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            # 如果是快速方案验证，则直接提取pkl文件
            if eval_types == ['proposal_fast']:
                result_file = result_file_path
                with open(result_file_path, 'rb') as f:
                    results = pickle.load(f)   
            # 创建coco对象: 基于val数据集
            coco = COCO(ann_file[1])
            gt_bboxes = []
            img_ids = coco.getImgIds()
            for i in range(len(img_ids)):
                ann_ids = coco.getAnnIds(imgIds = img_ids[i])  # 一个img_id对应多个ann_inds，这多个ann_inds会组成一个ann_info
                ann_info = coco.loadAnns(ann_ids)
                if len(ann_info) == 0: # 如果是空
                    gt_bboxes.append(np.zeros((0,4)))
                    continue
                # 提取每个img的ann中的信息
                bboxes=[]
                for ann in ann_info:
                    if ann.get('ignore', False) or ann['iscrowd']:
                        continue
                    x1,y1,w,h = ann['bbox']
                    bboxes.append([x1,y1,x1+w-1,y1+h-1])
                bboxes = np.array(bboxes, dtype = np.float32)  # 这里要转成ndarray做什么？
                if bboxes.shape[0] == 0:
                    bboxes = np.zeros((0,4))
                gt_bboxes.append(bboxes)
            # 计算recalls    
            max_dets = np.array([100, 300, 1000])   # 代表average recall的前100
            iou_thrs = np.arange(0.5, 0.96, 0.05)   # [0.5,0.55,0.60,...0.95]
            
            # 似乎这个proposal_fast的eval_recalls()只支持proposal格式[array1,...array5000],每个array(m,5)
            # 不支持propsal已经转换成按类分类的方式。
            
            recalls = eval_recalls(gt_bboxes, 
                                   results, 
                                   max_dets, 
                                   iou_thrs, 
                                   print_summary=False)
            avg_recall = recalls.mean(axis=1)       # 计算AR(average recall) (3,10)
            # 显示recall
            for i, num in enumerate(max_dets):
                print('AR@{}\t= {:.4f}'.format(num, avg_recall[i]))

            
            
            
            
            
    