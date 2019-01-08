#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:17:10 2018

@author: ubuntu
"""

from torch.nn.utils import clip_grad
from .hook import Hook
import os

class OptimizerHook(Hook):
    """优化器hook
    """
    def __init__(self, optimizer_config):
        self.grad_clip = optimizer_config['grad_clip']
        
    def write_txt(self,runner,results, file_name='record', type='a+'):
        """用于写入一个txt文件，默认路径在checkpoints文件夹
        写入模式：a+ 自由读写，扩展模式，文件没有就创建
        写入模式：w+ 自由读写，覆盖模式
        """
        import sys
        if sys.platform == 'linux':
            directory = '/home/ubuntu/suliang_git/slcv/checkpoints'   # for ubuntu 
        else:                     
            directory = '/Users/suliang/slcv/checkpoints/'            # for mac os    
        
        file_path = os.path.join(directory, file_name +'_epoch_{}'.format(runner._epoch))        
        with open(file_path, type) as f:  # 以结尾写入的方式打开，只有'a'和'at'两种模式的指针是在文件末尾
            print(results, file = f)
    
    def _output_grad_std(self, runner):    
        """输出grad_std，检查是否有梯度消失或梯度爆炸
        model.named_parameters()为一个迭代器,
        """
        if j%500==0 and j!=0:
            out=[]
        for k,(name, p) in enumerate(runner.model.named_parameters()):
            out.append((name, p.grad.data.std()))
        
        self.write_txt(out, file_name='grad',type='a+')
        
    def _output_param_std(self, runner):        
        """输出参数std，检查参数是否正常"""
        out = []
        for k,(name, p) in enumerate(runner.model.named_parameters()):
            out.append((name, p.data.mean(),p.data.std()))
        
        self.write_txt(out, file_name='parameter',type='a+')   
    
    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip)

    def after_train_iter(self, runner):
        """每个循环结束：优化器清零，基于损失计算梯度(loss.backward)，基于梯度用优化器更新参数(optimizer step)，"""
        runner.optimizer.zero_grad()            # 清零上一个iter的梯度
        runner.batch_output['loss'].backward()       # 计算梯度
#        if self.grad_clip is not None:
#            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()                 # 更新参数
        

if __name__ == '__main__':
    h = OptimizerHook()