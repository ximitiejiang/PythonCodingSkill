import copy
import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


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


class RepeatDataset(object):

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len


class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)


def get_dataset(data_cfg, dataset_class):
    """"获得数据集
    Args:
        data_cfg(dict): 存放所有数据集初始化参数的字典cfg.data.train
        dataset_class(class): 数据集的类
    Return:
        dset(obj): 生成的数据集
    """
    if data_cfg['type'] == 'RepeatDataset':
        return RepeatDataset(
            get_dataset(data_cfg['dataset'], dataset_class), data_cfg['times'])

    if isinstance(data_cfg['ann_file'], (list, tuple)):
        ann_files = data_cfg['ann_file']
        num_dset = len(ann_files)
    else:
        ann_files = [data_cfg['ann_file']]
        num_dset = 1

    if 'proposal_file' in data_cfg.keys():
        if isinstance(data_cfg['proposal_file'], (list, tuple)):
            proposal_files = data_cfg['proposal_file']
        else:
            proposal_files = [data_cfg['proposal_file']]
    else:
        proposal_files = [None] * num_dset
    assert len(proposal_files) == num_dset

    if isinstance(data_cfg['img_prefix'], (list, tuple)):
        img_prefixes = data_cfg['img_prefix']
    else:
        img_prefixes = [data_cfg['img_prefix']] * num_dset
    assert len(img_prefixes) == num_dset

    dsets = []
    for i in range(num_dset):
        data_info = copy.deepcopy(data_cfg)   # 需要深拷贝，是因为后边会pop()操作避免修改了原始cfg
        data_info['ann_file'] = ann_files[i]
        data_info['proposal_file'] = proposal_files[i]
        data_info['img_prefix'] = img_prefixes[i]
        
        data_info.pop('type')
        dset = dataset_class(**data_info)  # 弹出type字段，剩下就是数据集的参数
        dsets.append(dset)
    if len(dsets) > 1:
        dset = ConcatDataset(dsets)
    else:
        dset = dsets[0]
    return dset



if __name__ == '__main__':
    """总结绝对导入和相对导入：
    1. 带./..的都叫做相对导入，相对导入的好处是不受硬编码的影响，但缺点是不能直接运行
    2. 三种导入：隐式相对导入(from A import a)，显式相对导入(from .A import a)，绝对导入
    """
#    sys.path.append(os.path.abspath('..')) 
    import sys, os

    sys.path.insert(0, os.path.abspath('..'))
    print(sys.path)
    
    data = 'voc'

    
    if data == 'voc':
        from dataset.voc_dataset import VOCDataset
        data_root = '../data/VOCdevkit/'
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
        cfg_train=dict(
            type='RepeatDataset',
            times=10,
            dataset=dict(
                type='VOCDataset',
                ann_file=[
                    data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                    data_root + 'VOC2012/ImageSets/Main/trainval.txt'
                ],
                img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
                img_scale=(300, 300),
                img_norm_cfg=img_norm_cfg,
                size_divisor=None,
                flip_ratio=0.5,
                with_mask=False,
                with_crowd=False,
                with_label=True,
                test_mode=False,
                extra_aug=dict(
                    photo_metric_distortion=dict(
                        brightness_delta=32,
                        contrast_range=(0.5, 1.5),
                        saturation_range=(0.5, 1.5),
                        hue_delta=18),
                    expand=dict(
                        mean=img_norm_cfg['mean'],
                        to_rgb=img_norm_cfg['to_rgb'],
                        ratio_range=(1, 4)),
                    random_crop=dict(
                        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
                resize_keep_ratio=False))
        
        trainset = get_dataset(cfg_train, VOCDataset)
    
    if data == 'coco':
        from dataset.coco_dataset import CocoDataset
        data_root = '../data/VOCdevkit/'
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
        cfg_train=dict(
            type='RepeatDataset',
            times=5,
            dataset=dict(
                type='CocoDataset',
                ann_file=data_root + 'annotations/instances_train2017.json',
                img_prefix=data_root + 'train2017/',
                img_scale=(300, 300),
                img_norm_cfg=img_norm_cfg,
                size_divisor=None,
                flip_ratio=0.5,
                with_mask=False,
                with_crowd=False,
                with_label=True,
                test_mode=False,
                extra_aug=dict(
                    photo_metric_distortion=dict(
                        brightness_delta=32,
                        contrast_range=(0.5, 1.5),
                        saturation_range=(0.5, 1.5),
                        hue_delta=18),
                    expand=dict(
                        mean=img_norm_cfg['mean'],
                        to_rgb=img_norm_cfg['to_rgb'],
                        ratio_range=(1, 4)),
                    random_crop=dict(
                        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
                resize_keep_ratio=False))

        trainset = get_dataset(cfg_train, CocoDataset)
        
        
        sys.path.pop(0)