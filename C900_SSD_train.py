#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:42:09 2019

@author: ubuntu
"""

import logging
from torch.nn.parallel import DataParallel
from mmcv import Config
from mmcv.runner import Runner
import torch.distributed as dist
#from C950_SSD_core import Config
from C904_SSD_detector import OneStageDetector

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
    
def train():
    training =True
    
    cfg_path = 'C901_SSD_config_300_vgg16.py'
    cfg = Config.fromfile(cfg_path)
    
    distributed = False
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))
    
    # build detector
    model = OneStageDetector(cfg)
    model = DataParallel(model)
    
    trainset = []
    dataloader = []
    
    runner = Runner()
    if cfg.resume_from:  # 恢复训练
        runner.resume()
    elif cfg.load_from:  # 加载参数进行测试
        runner.load_checkpoint()
    else:                # 从新训练
        runner.run(dataloader)
    
    
if __name__ == '__main__':
    train()
    