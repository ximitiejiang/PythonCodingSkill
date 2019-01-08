#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:13:15 2018

@author: ubuntu
"""
import visdom
import numpy as np
from .logger_hook import LoggerHook

class LoggerVisdomHook(LoggerHook):
    
    def __init__(self, log_config):
        """输入为log_config dict: {}"""
        super().__init__(log_config)
        self.vis = visdom.Visdom(env='slcv')
        self.m=0
        self.n=0
        print('vis init finish!')
    
    def log(self,runner):
        """数据显示到终端"""
        num_x = runner._iter
        xlabel = 'n_iters'
        # loss曲线
        self.vis.line(X=[num_x],
                      Y=[runner.log_buffer.average_output['loss']],
                      opts=dict(markers=False,xlabel=xlabel,ylabel='loss'),
                      win='loss',
                      name='loss',
                      update='append')
        # acc_topk曲线
        self.vis.line(X=[num_x],
                      Y=[runner.log_buffer.average_output['acc_topk']],
                      opts=dict(markers=False,xlabel=xlabel,ylabel='acc'),
                      win='accuracy',
                      name='acc_top{}'.format(runner.cfg.topk),
                      update='append')
        # acc_top1曲线
        self.vis.line(X=[num_x],
                      Y=[runner.log_buffer.average_output['acc_top1']],
                      opts=dict(markers=False,xlabel=xlabel,ylabel='acc'),
                      win='accuracy',
                      name='acc_top1',
                      update='append')
        # mAP
        
        ''' TODO: lr curve 
        def log_lr(self, runner):
            self.vis.line(X=[num_x],
                  Y=
                  opts=dict(markers=False,xlabel=xlabel,ylabel='learning rate'),
                  win='lr',
                  name='learning rate',
                  update='append')
        '''
        
        
        ''' TODO: heatmap
        self.vis.heatmap(X=np.outer(np.arange(1, 6), np.arange(1, 11)),
                        opts=dict(columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                                  rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
                        colormap='Electric'))
        '''


