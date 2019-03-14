#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 08:46:28 2019

@author: ubuntu
"""

import numpy as np
import matplotlib.pyplot as plt
cimport numpy as np  # ?

cdef inline np.float32_t max(np.float32_t a, np.float32_t b):  # ?
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):  # ?
    return a if a <= b else b


def py_cpu_nms(dets, thresh):
    # dets:(m,5)  thresh:scaler
#    x1 = dets[:,0]
#    y1 = dets[:,1]
#    x2 = dets[:,2]
#    y2 = dets[:,3]
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:,0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:,1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:,2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:,3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]
    
#    scores = dets[:,4]
#    areas = (y2-y1+1) * (x2-x1+1)
#    index = scores.argsort()[::-1]
    cdef np.ndarray[np.float32_t, ndim=1] areas = (y2-y1+1) * (x2-x1+1)
    cdef np.ndarray[np.int_t, ndim=1]  index = scores.argsort()[::-1]    # can be rewriten
    
    keep = []
    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = np.zeros(ndets, dtype=np.int)
    
    cdef int _i, _j
    
    cdef int i, j
    
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    cdef np.float32_t w, h
    cdef np.float32_t overlap, ious
    
    j=0
    
#    while index.size >0:
#
#        i = index[0]       # every time the first is the biggst, and add it directly
#        keep.append(i)
#        
#        x11 = np.maximum(x1[i], x1[index[1:]])    # calculate the points of overlap 
#        y11 = np.maximum(y1[i], y1[index[1:]])
#        x22 = np.minimum(x2[i], x2[index[1:]])
#        y22 = np.minimum(y2[i], y2[index[1:]])
#        
#        w = np.maximum(0, x22-x11+1)    # the weights of overlap
#        h = np.maximum(0, y22-y11+1)    # the height of overlap
#       
#        overlaps = w*h
#        
#        ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
#        
#        idx = np.where(ious<=thresh)[0]
#        
#        index = index[idx+1]   # because index start from 1    
#    return keep
    for _i in range(ndets):
        i = index[_i]
        
        if suppressed[i] == 1:
            continue
        keep.append(i)
        
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        
        iarea = areas[i]
        
        for _j in range(_i+1, ndets):
            j = index[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = max(ix2, x2[j])
            yy2 = max(iy2, y2[j])
    
            w = max(0.0, xx2-xx1+1)
            h = max(0.0, yy2-yy1+1)
            
            overlap = w*h 
            ious = overlap / (iarea + areas[j] - overlap)
            if ious>thresh:
                suppressed[j] = 1
    
    return keep    

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

