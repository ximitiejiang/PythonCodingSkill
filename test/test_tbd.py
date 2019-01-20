#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:10:14 2019

@author: ubuntu
"""

import torch.nn as nn
class Registry(object):
    def __init__(self):
        print('this is init')
        self.module_dict={}
    
    def __call__(self, class_type):
        print('this is call')
        module_name = class_type.__name__
        self.module_dict[module_name] = class_type
        return class_type
    
#    def __init__(self, name):
#        self._name = name
#        self._module_dict = dict()
#    def _register_module(self, module_class):
#        if not issubclass(module_class, nn.Module):
#            raise TypeError(
#                'module must be a child of nn.Module, but got {}'.format(
#                    type(module_class)))
#        module_name = module_class.__name__
#        if module_name in self._module_dict:
#            raise KeyError('{} is already registered in {}'.format(
#                module_name, self.name))
#        self._module_dict[module_name] = module_class
#    def register_module(self, cls):  # 装饰器函数：传入一个类cls，返回一个类cls
#        self._register_module(cls)
#        return cls
        
#    def register_module(self, class_type):
#        module_name = class_type.__name__
#        self._module_dict[module_name] = class_type
#        return class_type
        
#backbones = Registry()  # 创建一个Registry对象
#@backbones.register_module         # 挂一个装饰器：用对象的方法作为装饰器，传入的是一个类名，比如ResNet
registry = Registry()        

@registry
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
    def forwrad(self):
        pass
print(registry.module_dict)
#model = ResNet()
print(registry.module_dict)
