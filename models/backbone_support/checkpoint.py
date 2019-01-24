#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 09:51:19 2019

@author: suliang
"""
import os
import torch
from collections import OrderedDict
from torch.utils import model_zoo


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.
    Args:
        module(nn.Module type)
        state_dict(Ordereddict)
        strick(bool)
    """
    # 存放额外多出来的key
    unexpected_keys = [] 
    
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            unexpected_keys.append(name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        try:
            own_state[name].copy_(param)  # 拷贝checkpoint的参数到model中
        except Exception:
            raise RuntimeError('While copying the parameter named {}, '
                               'whose dimensions in the model are {} and '
                               'whose dimensions in the checkpoint are {}.'
                               .format(name, own_state[name].size(),
                                       param.size()))
    # 存放少了的key
    missing_keys = set(own_state.keys()) - set(state_dict.keys())

    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))
    err_msg = '\n'.join(err_msg)
    if err_msg:
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warn(err_msg)
        else:
            print(err_msg)
    

def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    """加载checkpoint，把state_dict传递给model，并返回checkpoint字典(可用于提取checkpoint中其他信息)
    Args：
        filename(str): 
        map_location(): 可以选择''
        strict(是否允许不同参数)
    Return:
        checkpoint(dict)
    torch.load()参考：https://pytorch.org/docs/stable/torch.html
    to same cpu or GPU: torch.load('gen_500000.pkl')
    to->cpu: torch.load('gen.pkl', map_location=lambda storage, loc: storage)
    cpu->GPU(1): torch.load('gen.pkl', map_location=lambda storage, loc: storage.cuda(1))
    GPU0->GPU1: torch.load('gen.pkl', map_location={'cuda:0':'cuda:1'})
    """
    if filename.startswith(('http://', 'https://')):
        checkpoint = model_zoo.load_url(filename)  # 借用pytorch的model_zoo:如果文件存在于torch_home则直接load, 否则下载    
    elif not os.path.isfile(filename):
        raise IOError('{} is not a checkpoint file'.format(filename))
    else:
        # 加载checkpoint
        print('loading checkpoint file: {}'.format(filename))
        checkpoint = torch.load(filename, map_location = map_location)
    # ----------------从checkpoint获得state_dict-------------------
    if isinstance(checkpoint, OrderedDict): # 如果直接存的是OrderedDict则直接读取
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: # 如果存的是dict则读取state_dict
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # 如果是data paralle model，则去掉module关键字(即从第7个字符开始)后得到state_dict
    # 参考：https://www.ptorch.com/news/74.html
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}        
    # --------------把state_dict加载到模型中-------------------------
    if hasattr(model, 'module'):  # 如果是data paralle model，则需要提取module再加载state_dict
        load_state_dict(model.module, state_dict, logger)
    else:  # 如果是普通模型，则直接用model加载state_dict
        load_state_dict(model, state_dict, logger)
    # model虽已加载state_dict,返回checkpoint中还包含的其他信息     
    return checkpoint


def save_checkpoint(self, out_dir, filename, save_optimizer=True, meta=None):
    """保存checkpoint到文件
    输入：meta 保存version和time, dict, 默认是{'epoch':epoch, 'iter':iter}
          out_dir保存地址
          filename保存文件名
          save_optimizer是否保存优化器        runner.save_checkpoint(
        self.out_dir, runner.cfg.model_name, save_optimizer=self.save_optimizer, **self.args)

    输出：OrderedDict {'meta':dict, 'state_dict':OrderedDict, 'optimizer':dict}
    """
    if meta is None:
        meta = dict(epoch=self._epoch +1, iter = self._iter)
    else:
        meta.update(epoch=self._epoch +1, 
                    iter = self._iter,
                    time = time.time())
    # 判断是否保存optimizer
    appendname = '_epoch_{}.pth'
    filepath = os.path.join(out_dir, filename + appendname.format(self._epoch+1))
    optimizer = self.optimizer if save_optimizer else None
    # 从GPU拷贝state_dict到cpu
    state_dict_cpu = OrderedDict()
    for key, val in self.model.state_dict().items():
        state_dict_cpu[key] = val.cpu()
#            state_dict_cpu[key] = val
    # 生成checkpoint    
    checkpoint ={'meta': meta,
                 'state_dict': state_dict_cpu}
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    # 保存checkpoint
    torch.save(checkpoint, filepath)



if __name__=='__main__':
    pass