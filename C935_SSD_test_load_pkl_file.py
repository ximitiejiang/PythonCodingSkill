#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:29:43 2019

@author: ubuntu
"""
#from six.moves import cPickle as pickle
import _pickle as pickle
"""注意cPickle, Pickle, six.moves的区别：
1. cPickle是c代码写成，Pickle是python写成，相比之下cPickle更快
2. cPickle只在python2中存在，python3中换成_pickle了
3. six这个包是用来兼容python2/python3的，这应该是six的由来(是2与3的公倍数)
   six包里边集成了有冲突的一些包，所以可以从里边导入cPickle这个在python3已经取消的包
"""

def evaluation(result_file_path):
    """基于已经生成好的pkl模型预测文件，进行相关操作
    """
    pass

if __name__=='__main__':
    eval_with_pkl_file = True
    # 打开pkl获得数据
    if eval_with_pkl_file:
        result_file_path = 'dataset_eval_result/results.pkl'
        with open(result_file_path, 'r') as f:
            results = pickle.load(f)
    # 调用coco_eval()即结束(内部再调用fast_eval_recall(result_file, coco, max_dets))
    # 内部再调用eval_recalls()
    # 
    """
    gts(list): [array1, ...arrayn], 5000个array, 每个array相当与一张图，包含(m,4)个bboxes
    proposals([lst1,...lstn]), 5000个lst，每个list相当于一张图，每个list包含80个array, 
        所以proposal[0][0]就是第0张图的第0个类的预测结果，每个类的预测结果为(n, 5)
    proposal_nums(ndarray) ：默认为[100,300,1000]
    """
    