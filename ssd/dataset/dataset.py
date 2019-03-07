#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 23:26:11 2019

@author: suliang
"""
# %% 
"""Q.数据集的图片，第一步是显示，如何是最优显示？
1. plt.imshow()
   cv2.imshow()
2. 
"""
# 
import matplotlib.pyplot as plt
 
def color2value(color_str):
    """定义一个把颜色字符串转换成opencv能识别的tuple"""
    colors = dict(red = (0, 0, 255),
                  green = (0, 255, 0),
                  blue = (255, 0, 0),
                  cyan = (255, 255, 0),
                  yellow = (0, 255, 255),
                  magenta = (255, 0, 255),
                  white = (255, 255, 255),
                  black = (0, 0, 0))
    return colors[color_str]

def imshow_bboxes(img, bboxes, 
                  colors='green',
                  thickness=1):
    """用来显示img和bboxes
    注意颜色需要传入len=3的list/tuple，不能是字符str
    Args:
        img(ndarray): (h,w,c)/bgr 代表已读入的图片，为bgr颜色空间, 通道顺序(h,w,c), 内部会转成rgb来显示
        bboxes(ndarray): (m,4)代表bboxes坐标,[xmin,ymin,xmax,ymax]
        colors(str/list): 代表bboxes的颜色，支持每个bbox显示不同颜色
    """
    if not isinstance(bboxes, np.ndarray):      # 如果是直接读取img为ndarray，否则
        bboxes = np.array(bboxes)                   # ndarray to list
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [color2value(color) for color in colors]  # 转换字符串成tuple
    for i, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.int32)
        lt = (bbox[0], bbox[1])
        rb = (bbox[2], bbox[3])
        cv2.rectangle(img,lt,rb,colors[i],thickness=thickness)
    plt.imshow(img[...,[2,1,0]])    
    

def imshow_bboxes_labels(img, bboxes, labels, score_thr=0.3,
                         class_names=None,
                         bbox_colors='green',
                         text_colors='green',
                         thickness=1,
                         font_scale=0.5):
    """用来显示img和bboxes和labels
    可用来显示train/val的现成img/bbox/label，此时bbox为4列坐标[xmin,ymin,xmax,ymax]
    也可以用来显示test的预测结果img/bbox/label，此时bbox为5列坐标，第5列为置信度 [xmin,ymin,xmax,ymax,score]
    Args:
        img(ndarray): (h,w,c)/bgr 代表已读入的图片，为bgr颜色空间, 通道顺序(h,w,c), 内部会转成rgb来显示
        bboxes(ndarray): (m,4),或(m,5)代表bboxes坐标,[xmin,ymin,xmax,ymax, /score]
        labels(ndarray/list): (m,)代表每个对应bbox的label
        score_thr(float): 代表预测的bbox的置信度，=0表示没有置信度的4列坐标，>0表示有置信度的5列坐标
        class_names(list/tuple): 代表跟label编号对应的类型字符串列表
        bbox_colors(str/list): 代表bboxes的颜色，支持每个bbox显示不同颜色
        text_colors(str/list): 代表文字颜色，支持每种label显示不同颜色
        font_scale(float): 代表字体缩放比例
    """
    if not isinstance(bboxes, np.ndarray):      # 如果是直接读取img为ndarray，否则
        bboxes = np.array(bboxes)                   # ndarray to list
    if not isinstance(bbox_colors, list):
        bbox_colors = [bbox_colors for _ in range(len(bboxes))]
    if not isinstance(text_colors, list):
        text_colors = [text_colors for _ in range(len(bboxes))]
    bbox_colors = [color2value(color) for color in bbox_colors]  # 转换字符串成tuple
    text_colors = [color2value(color) for color in text_colors]  # 转换字符串成tuple
    if score_thr > 0: # 只显示置信度大于阀值的
        scores = bboxes[:,-1]
        score_id = scores > score_thr
        # 获得scores, bboxes
        scores = scores[score_id]
        bboxes = bboxes[score_id]
    # 获得label_text
    if class_names is not None:
        label_text = [class_names[label] for label in labels]
    else:
        label_text = ['cls {}'.format(label) for label in labels]
    # 循环显示添加到img上
    for i, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.int32)
        lt = (bbox[0], bbox[1])
        rb = (bbox[2], bbox[3])
        cv2.rectangle(img, lt, rb, bbox_colors[i], thickness=thickness)
        
        if len(bbox) > 4:
            label_text += '|{:.2f}'.format(bbox[-1])
        cv2.putText(img, label_text[i], (bbox[0],bbox[1]-2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_colors[i])
    plt.imshow(img[...,[2,1,0]])


def vis_bbox(img, bbox, label=None, score=None, score_thr=0, label_names=None,
             instance_colors=None, alpha=1., linewidth=1.5, ax=None):
    """另外一个图片+bbox显示的代码，感觉效果比cv2的好上几条街(来自chainercv)
    对应数据格式和注释已修改为匹配现有voc/coco数据集。
    Args:
        img (ndarray): (h,w,c), BGR and the range of its value is
            :math:`[0, 255]`. If this is :obj:`None`, no image is displayed.
        bbox (ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(x_{min}, y_{min}, x_{max}, y_{max})` in the second axis.
        label (ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        score_thr(float): A float in (0, 1), bboxes scores with lower than
            score_thr will be skipped. if 0 means all bboxes will be shown.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        instance_colors (iterable of tuples): List of colors.
            Each color is RGB format and the range of its values is
            :math:`[0, 255]`. The :obj:`i`-th element is the color used
            to visualize the :obj:`i`-th instance.
            If :obj:`instance_colors` is :obj:`None`, the red is used for
            all boxes.
        alpha (float): The value which determines transparency of the
            bounding boxes. The range of this value is :math:`[0, 1]`.
        linewidth (float): The thickness of the edges of the bounding boxes.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    from matplotlib import pyplot as plt        
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')
    
    if score_thr > 0:                      # 只显示置信度大于阀值的bbox
        score_id = score > score_thr
        # 获得scores, bboxes
        score = score[score_id]
        bbox = bbox[score_id]
        label = label[score_id]
        
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    if img is not None:
        img = img[...,[2,1,0]]         # hwc/bgr to rgb
        ax.imshow(img.astype(np.uint8))
    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    if instance_colors is None:
        # Red
        instance_colors = np.zeros((len(bbox), 3), dtype=np.float32)
        instance_colors[:, 0] = 255
    instance_colors = np.array(instance_colors)

    for i, bb in enumerate(bbox):        # xyxy to xywh
        xy = (bb[0], bb[1])
        height = bb[3] - bb[1]
        width = bb[2] - bb[0]
                
        color = instance_colors[i % len(instance_colors)] / 255
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False,
            edgecolor=color, linewidth=linewidth, alpha=alpha))

        caption = []
        if label is not None and label_names is not None:
            lb = label[i]
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))
        if len(caption) > 0:
            ax.text(bb[0], bb[1],
                    ': '.join(caption),
                    style='italic',
                    color = 'b',  # 默认是黑色，这里设为blue
                    bbox={'facecolor': 'white', 'alpha': 0.3, 'pad': 1}) 
                    #文字底色：白色，透明度0.2，边空1
    return ax


# %% 
"""Q.数据集类的基本结构和原理？
1. 数据集必须强制包含__getitem__和__len__因为这是在实现切片和合并数据源必要的操作
2. 数据集通常只需要传入ann_file地址，就能够完成数据集的创建，coco/voc都是这样
"""
# 参考pytorch的基础类Dataset
class Dataset(object):
    """该类来自torch.utils.data.Dataset
    基础数据集类，增加重载运算__add__对数据集进行叠加
    """
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


# %%
"""Q.为什么数据集能够组合？如何实现？
"""
import bisect
class ConcatDataset(Dataset):
    """该类来自torch.utils.data.ConcatDataset
    确定个数的不同来源的数据集堆叠(class from pytorch)，可用于如voc07/12的组合
    Args:
        datasets(list): [dset1, dset2..] 代表待叠加的数据集
    """
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)  #得到list(len(p1), len(p1+p2), len(p1+p2+p3),...)

    def __len__(self):
        """把ConcatDataset的len函数设计成list,包含每个dataset的长度累加值，
        更便于频繁操作__getitem__函数的编写(__getitem__要尽可能高效)
        """
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx) #先得到数据集编号
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1] #再得到样本编号
        return self.datasets[dataset_idx][sample_idx]


# %%
"""Q.如何读取VOC dataset？
"""
from torch.utils.data import Dataset
import cv2
import xml.etree.ElementTree as ET
from addict import Dict
import numpy as np


class VOCDataset(Dataset): # 继承Dataset的好处是可以使用他的__add__方法以及该方法的ConcateDataset()类
    """VOC数据集类：包含__getitem__, __len__的pytorch数据类
    数据读取过程：图片名称检索文件(txt) -> 所有图片基础信息(xml) -> 图片信息(jpg)
    最简方式只要提供ann_file路径，就能创建一个基础版dataset
    注意1：xml文件格式如下
    -<annotation>
        -<size>
            <width>500</width>
            <height>375</height>
            <depth>2<depth>
         </size>
        -<object>
            <name>chair</name>
            <difficlut>0</difficult>
           -<bndbox>
               <xmin>263</xmin>
               <ymin>211</ymin>
               <xmax>324</xmax>
               <ymax>339</ymax>
            </bndbox>
         <object>
    -</annotation>
    注意2：voc数据集每个bbox有一个difficult标志，代表难度大的bbox，一张图片可能同时存在
    多个difficult=0的低难度框和difficult=1的高难度框(比如00005)
    1. 数据集：该数据集包的子文件夹：
        (1)Annotations文件夹: 包含图片bbox等信息的xml文件
        (2)JPEGImages文件夹: 包含图片jpg文件
        (3)ImageSets文件夹: 包含图片文件名检索文件(起始步)
        (4)labels: 略
        (5)SegmentationClass: 略
        (6)SegmentationObject: 略
    2. 该类继承自Dataset的好处是可以使用Dataset自带的__add__方法以及该方法引入的ConcateDataset()类
    用来叠加一个数据集的多个子数据源，比如voc07和voc12

    Args:
        ann_file(list): ['ann1', 'ann2'..] 代表图片名检索文件，可以包含1-n个数据源的不同检索文件
        img_prefix(list): 代表检索文件名的前缀，前缀+检索文件名 = 完整文件名    
    """
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
    
    def __init__(self, ann_file, img_prefix):
        self.img_prefix = img_prefix
        self.cat2label = {name: label for label, name in enumerate(self.CLASSES)}
        # 从图片名检索文件读取所有图片名称(txt文件的处理)
        self.img_ids = []
        with open(ann_file) as f:
            lines = f.readlines()
            for line in lines:
                self.img_ids.append(line.strip('\n')) # 
        
    def __getitem__(self, idx):
        # 从idx转化成path路径
        img_id = self.img_ids[idx]
        img_path = self.img_prefix + 'JPEGImages/' + '{}.jpg'.format(img_id)
        xml_path = self.img_prefix + 'Annotations/' + '{}.xml'.format(img_id)
        # 读取指定img图片 (jpg文件的处理)
        img = cv2.imread(img_path)        # (h,w,c)/bgr
        # 读取指定img相关数据 (xml文件的处理)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            bnd_box = obj.find('bndbox')
            bbox = [int(bnd_box.find('xmin').text),
                    int(bnd_box.find('ymin').text),
                    int(bnd_box.find('xmax').text),
                    int(bnd_box.find('ymax').text)]
            name = obj.find('name').text
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            if difficult:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        # 补齐bboxes/labels
        if not bboxes:       # 如果没有difficult=0的数据，则创建空数组
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2)  # 同时把list转换成array
            labels = np.array(labels)
        if not bboxes_ignore: # 如果没有difficult=1的数据，则创建空数组
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)  # 同时把list转换成array
            labels_ignore = np.array(labels_ignore)
        # 转换数据格式适配pytorch，捆绑数据
        img_data = Dict(img = img,
                        width = width,
                        height = height,
                        bboxes = bboxes.astype(np.float32),
                        labels = labels.astype(np.int64),
                        bboxes_ignore = bboxes_ignore.astype(np.float32),
                        labels_ignore = labels_ignore.astype(np.int64))
        return img_data
        
    def __len__(self):
        return len(self.img_ids)

if __name__ =='__main__':
    run_voc=True
    if run_voc:
        data_root = 'data/VOCdevkit_mac/'  # 如果是mac则增加_mac后缀
        ann_file=[data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                  data_root + 'VOC2012/ImageSets/Main/trainval.txt']
        img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/']
        
        voc07 = VOCDataset(ann_file[0], img_prefix[0])
        voc12 = VOCDataset(ann_file[0], img_prefix[0])
        dataset = voc07 + voc12             # Dataset类有重载运算符__add__，所以能够直接相加 (5011+5011)
        classes = voc07.CLASSES
        img_data = dataset[9078]               # len = 10022
#        imshow_bboxes_labels(img_data.img, img_data.bboxes, img_data.labels,
#                             class_names = classes)
        """img(ndarray/chw/rgb), bbox()"""
        vis_bbox(img_data.img, img_data.bboxes, label=img_data.labels, score=None, label_names=classes,
                 instance_colors=None, alpha=1., linewidth=1.5, ax=None)
        plt.show()


# %%
"""Q.如何读取COCO dataset？
"""
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import os

class CocoDataset(Dataset):
    """Coco数据集类
    注意1：需要预装coco api
    注意2：coco数据集有一个iscrowd标志，=0代表该annotation实例是单个对象，将在segmentation
    处使用polygons格式给出，而如果=1代表该annotation实例为一组对象，将使用RLE格式给出
    注意2：json文件结构：5段式(info/licenses/images/annotations/categories)
    {
         "info": info,
         "licenses": [license],
         "images": [image],
         "annotations": [annotation]
         "categories": [category]
    }
    例如：如下是对基本的coco instance的json文件的实例
    "info":{
            "description":"This is stable 1.0 version of the 2014 MS COCO dataset.",
            "url":"http:\/\/mscoco.org",
            "version":"1.0","year":2014,
            "contributor":"Microsoft COCO group",
            "date_created":"2015-01-27 09:11:52.357475"
            },
    images{
            "license":3,
            "file_name":"COCO_val2014_000000391895.jpg",                        # 重要(图片名称)
            "coco_url":"http:\/\/mscoco.org\/images\/391895",
            "height":360,"width":640,"date_captured":"2013-11-14 11:18:45",     # 重要
            "flickr_url":"http:\/\/farm9.staticflickr.com\/8186\/8119368305_4e622c8349_z.jpg",
            "id":391895                                                         # 重要(ann id)
            },
    licenses{
            "url":"http:\/\/creativecommons.org\/licenses\/by-nc-sa\/2.0\/",
            "id":1,
            "name":"Attribution-NonCommercial-ShareAlike License"
            },
    annotation{
            "id": int,
            "image_id": int,
            "category_id": int,
            "segmentation": RLE or [polygon],
            "area": float,
            "bbox": [x,y,width,height],         # 重要 (xmin,ymin,w,h)
            "iscrowd": 0 or 1,
            },
    categories{
            "supercategory": "vehicle",         # 重要
            "id": 2,
            "name": "bicycle"                   # 重要
            }
    1. 该数据集包含文件夹: 很简单，很直接
        (1) annotations: 标注文件 (里边有instances/实例, captions/标题说明(用于看图说话), keypoints/关键点, 这3类标注)
        (2) train2017: 训练图片
        (3) val2017: 验证图片
        (4) test2017: 测试图片
    2. coco api的几个核心函数：如下6个函数就满足coco数据集处理需求了
        (1) 创建coco对象：
            coco = COCO(ann_file)
        (2) 获得：
            coco.getCatIds(): 获得所有分类Ids, 这个Ids跟80个分类字符串一一对应，从0开始但不是连续的
            coco.getImgIds(): 获得所有图片Ids，每个图片id为一串6位数字
            coco.getAnnIds(): 通过输入imgId获得对应AnnId
        (3) 加载：
            coco.loadImgs(img_id): 通过输入img_id获得img_info
            coco.loadAnns(ann_id): 通过输入ann_id获得ann_info
    Args:
        ann_file(list): ['ann1', 'ann2'..] 代表图片名检索文件，可以包含1-n个数据源的不同检索文件
        img_prefix(list): 代表检索文件名的前缀，前缀+检索文件名 = 完整文件名    
    """
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
    
    def __init__(self, ann_file, img_prefix):
        self.img_prefix = img_prefix
        
        # 获得图片名称：先安装coco api (json文件处理)
        self.coco = COCO(ann_file)            # 总共80个大类(但注意编号是从1-90且不连续，中间有数字缺),不是指超类(超类更少)
        self.cat_ids = self.coco.getCatIds()  # 获得分类对应的值
        self.cat2label = {id: i for i,id in enumerate(self.cat_ids)} # 此处做了修改把id:i+1改成了id:i否则标签错位
        
        self.img_ids = self.coco.getImgIds()  # 获得所有图片ids: 总计118287张图(大约是voc的20倍): 用img_id -> img_info和ann_id
        self.img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            self.img_infos.append(info)
        
    def __getitem__(self, idx):
        img_info = self.img_infos[idx]
        # 读取图片：
        img_path = os.path.join(self.img_prefix, img_info['file_name'])
        img = cv2.imread(img_path)
        # 读取图片辅助信息: 先要获得img_id, 然后才能得到ann_id, 然后才能得到ann_info
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        # 解析ann info: 得到的ann_info包含多个对象的ann_info, 当成dict操作
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []   # 多对象放在ignore，一般用来做segmentation，不用label_ignore?
        # 多对象数据放在ignore里，做物体检测先只考虑iscrowd=0的单对象情况
        for i, ann in enumerate(ann_info):
            x1,y1,w,h = ann['bbox']
            bbox = [x1,y1,x1+w,y1+h]  #
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
        # TODO: 暂时没有考虑mask数据的加入        
        
        # 补齐bboxes/labels
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32) 
        
        img_data = Dict(img = img,
                        bboxes = gt_bboxes.astype(np.float32),
                        labels = gt_labels.astype(np.int64),
                        bboxes_ignore = gt_bboxes_ignore.astype(np.float32))

        return img_data

    def __len__(self):
        pass

if __name__=="__main__":    
    run_coco = False
    if run_coco:
        data_root = 'data/coco/'    # 需要预先把主目录加进sys.path
        ann_file=[data_root + 'annotations/instances_train2017.json',
                  data_root + 'annotations/instances_val2017.json']
        img_prefix=[data_root + 'train2017/', data_root + 'val2017/']
        
        dataset = CocoDataset(ann_file[0], img_prefix[0])
        classes = dataset.CLASSES
        img_data = dataset[8]
            
        imshow_bboxes_labels(img_data.img, img_data.bboxes, img_data.labels,
                             class_names = classes)

# %%
"""Q.如果只做车辆和行人检测，如何从coco数据集分离出车辆和行人数据用来进行训练？
"""
class Coco_vehicle_person(CocoDataset):
    """应该只需要重写init，保证img_ids读入的是需要的分类就可以了
    """
    def __init__(self, ann_file, img_prefix):
        pass


        
        
# %%
"""对于图像的transform，有哪些方法可用？
1. 读入图像：img = cv2.imread(path)                     - (h,w,c)/(bgr 0~255)/ndarray
2. 显示图像：plt.imshow(img)，注意只能显示rgb             - (h,w,c)/(bgr 0~255)/ndarray
3. 额外扩增：                                           - (h,w,c)/(bgr 0~255)/ndarray
4. 缩放或尺寸变换：                                      - (h,w,c)/(bgr 0~255)/ndarray
5. 归一化：                                            - (h,w,c)/(bgr -2.x~2.x)/ndarray
6. bgr2rgb                                            - (h,w,c)/(rgb -2.x~2.x)/ndarray
7. padding
8. flip或rotate
9. transpose                                          - (c,h,w)/(rgb -2.x~2.x)/ndarray
10. to tensor                                         - (h,w,c)/(rgb -2.x~2.x)/tensor

"""



# %%
"""Q.如何做尺寸变换和尺寸缩放？

"""

#cv2.resize(img, size, interpolation=interp_codes[interpolation])
def imresize():
    pass


def imrescale():
    pass
    