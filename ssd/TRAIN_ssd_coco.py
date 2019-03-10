#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:42:09 2019

@author: ubuntu
"""

import logging
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader

import torch.distributed as dist
from collections import OrderedDict
import torch
from mmcv.runner import Runner

from utils.config import Config
from model.one_stage_detector import OneStageDetector
from dataset.voc_dataset import VOCDataset
from dataset.utils import get_dataset

def get_dist_info():
    if dist._initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    return logger

def batch_processor(model, data, train_mode):
    """创建一个基础batch process，用来搭配runner模块进行整个计算框架的组成
    1. 计算损失
    2. 解析损失并组合输出
    Args:
        model(Module)
        data()
    Returns:
        
    """
    losses = model(**data)
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))
    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    
    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()
        
    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs  
  
def train():
    training =True
    
    # get cfg
    cfg_path = 'config/cfg_ssd300_vgg16_voc.py'
    cfg = Config.fromfile(cfg_path)
    
    # set backends
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    # get logger
    distributed = False
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))
    
    # build model & detector
    model = OneStageDetector(cfg)
    model = DataParallel(model)
    
    # prepare data & dataloader
    dataset = get_dataset(cfg.data.train, VOCDataset)
    batch_size = cfg.gpus * cfg.data.imgs_per_gpu
    num_workers = cfg.gpus * cfg.data.workers_per_gpu
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers)

    # define runner and running type(1.resume, 2.load, 3.train/test)
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir, cfg.log_level)
    if cfg.resume_from:  # 恢复训练
        runner.resume(cfg.resume_from)
    elif cfg.load_from:  # 加载参数进行测试
        runner.load_checkpoint(cfg.load_from)
    else:                # 从新训练
        runner.run(dataloader, cfg.workflow, cfg.total_epochs)
    
    
if __name__ == '__main__':
    train()
    