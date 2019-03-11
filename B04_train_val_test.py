#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:26:35 2019

@author: ubuntu
"""
# %%
"""如何用训练框架类Runner?
1. 创建
2. 
"""
from mmcv import Runner
# 
def main():
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir, cfg.log_level)
    if cfg.resume_from:  # 恢复训练
        runner.resume(cfg.resume_from)
    elif cfg.load_from:  # 加载参数进行测试
        runner.load_checkpoint(cfg.load_from)
    else:                # 从新训练
        runner.run(dataloader, cfg.workflow, cfg.total_epochs)

if __name__ == '__main__':
    main()

# %%
"""在Runner中如何创建optimizer?
"""





# %%
"""Q. 如何设计一个config类，方便从config文件中调用参数？
""" 
from addict import Dict
class Config():
    """定义Config类，可以从.py文件获得参数，并能以简洁方式取得变量/字典的值"""
    def __init__(self, config_path):
        pass
    def __getattr__(self,name):
        pass



# %%
"""Q. 如何设计简化版runner类，便于进行训练？
1. 集成了train/val/test
2. 
"""
class Runner():
    def __init__(self):
        pass
    def train(self):
        pass
    def val(self):
        pass



# %% 
"""Q. 如何设计hook系统？
"""


# %%
"""Q. 如何设计
"""
