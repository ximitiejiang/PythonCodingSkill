#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:03:28 2018

@author: ubuntu

实践一下自己的框架：基于trainer类进行迁移学习

"""

#from torch.utils.data import DataLoader
#from slcv.runner.runner import Runner
#from slcv.cfg.config import Config
#import torch
#from slcv.model.pretrained_models import pretrained_models
#from slcv.dataset.dogcat import DogCat

from torchvision import models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import Runner
from dogcate_dataset import DogcatDataset


def main():
    cfg_path = './cfg_resnet50_dogcat.py'
    cfg = Config.fromfile(cfg_path)
    
    # build model
    model = models.resnet50(pretrained=True)
    
    # frozen pretrained params's grads
    for param in model.parameters():
        param.requires_grad = False
    
    # modify last layer output classes from 1000 classes to 2 classes
    num_classes = 2
    fc_input_layers = model.fc.in_features
    model.fc = nn.Linear(fc_input_layers, num_classes, bias=True)
    
    # prepare data & dataloader
    trainset = DogcatDataset(cfg.train_root, transform=None, train=True, test=False)
    trainloader = DataLoader(trainset, batch_size = cfg.batch_size, 
                             shuffle = True, num_workers = 2)
    
    # 2. 模型
    if torch.cuda.device_count() > 0 and len(cfg.gpus) == 1:
        model = model.cuda()
    elif torch.cuda.device_count() > 1 and len(cfg.gpus) > 1:  # 数据并行模型
        model = torch.nn.DataParallel(model, device_ids=cfg.gpus).cuda()
    
    # 3. 训练
    runner = Runner(trainloader, model, cfg.optimizer, cfg) # cfg对象也先传进去，想挂参数应该是需要的
    runner.register_hooks(
            cfg.optimizer_config,
            cfg.checkpoint_config,
            cfg.logger_config
            )
    # 恢复训练
    if cfg.resume_from is not None:
        runner.resume(cfg.resume_from, resume_optimizer=True, map_location='default')  # 确保map_location与cfg的定义一致
    # 加载模型做inference
    elif cfg.load_from is not None:
        runner.load_checkpoint(cfg.load_from)
    
    runner.train()

if __name__=='__main__':
    main()        

