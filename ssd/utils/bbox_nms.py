import torch
from . import nms_wrapper


def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0], ), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels


def py_cpu_nms(dets, thresh):
    """python版本的cpu_nms:
    待验证是否可以在ssd采用该cpu版本nms计算，速度上是否影响很大    
    """
    # dets:(m,5)  thresh:scaler
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    
    areas = (y2-y1+1) * (x2-x1+1)
    scores = dets[:,4]
    keep = []
    
    index = scores.argsort()[::-1]  #因为-1反排，所以是从大到小的index
    while index.size >0:  # 每次循环更新index，
        i = index[0]       # 先取出最大置信度的bbox，直接放入keep
        keep.append(i)
        
        x11 = np.maximum(x1[i], x1[index[1:]])    # 计算其他所有bbox跟最大置信度bbox的交集方框对应的(x1,y1,x2,y2)
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        
        w = np.maximum(0, x22-x11+1)    # the weights of overlap
        h = np.maximum(0, y22-y11+1)    # the height of overlap
       
        overlaps = w * h
        ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
        idx = np.where(ious<=thresh)[0]    # 把ious小于thr的idx记录
        
        index = index[idx+1]   # because index start from 1
    return keep


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    boxes=np.array([[100,100,210,210,0.72],
                    [250,250,420,420,0.8],
                    [220,220,320,330,0.92],
                    [100,100,210,210,0.72],
                    [230,240,325,330,0.81],
                    [220,230,315,340,0.9]]) 
    def plot_bbox(dets, c='k'):
        x1 = dets[:,0]
        y1 = dets[:,1]
        x2 = dets[:,2]
        y2 = dets[:,3]
        plt.plot([x1,x2], [y1,y1], c)
        plt.plot([x1,x1], [y1,y2], c)
        plt.plot([x1,x2], [y2,y2], c)
        plt.plot([x2,x2], [y1,y2], c)
        plt.title("after nms")

    plot_bbox(boxes,'k')   # before nms
    keep = py_cpu_nms(boxes, thresh=0.7)
    plot_bbox(boxes[keep], 'r')# after nms