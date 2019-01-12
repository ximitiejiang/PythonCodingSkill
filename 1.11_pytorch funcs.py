#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 09:49:15 2019

@author: suliang
"""


'''-----------------------------data-------------------------------------------
Q. tensor的创建
'''
import torch
f1 = torch.tensor([[1,2,3],[4,5,6]])  # 默认float
f2 = torch.tensor(2,3)
f3 = torch.IntTensor([1,2,3])  # 整数tensor

torch.ones(2,3)
torch.zeros(2,3)
torch.eye(2,3)


'''------------------------------------------------------------------------
Q. tensor的数据格式转换
'''
import torch
import numpy as np

# 转成tensor
a = [1,2,3]
b = np.array([3,2,1])
c = dict(a=1,b=2)
torch.tensor(a)  # list转tensor
torch.tensor(b)  # array转tensor
torch.tensor(c)  # dict不可以转tensor

# tensor转其他
t1 = torch.tensor([1,2,3])
t2 = t1.numpy()             # tensor转numpy
t3 = t1.numpy().tolist()    # tensor转numpy,numpy再转list

b0 = torch.tensor(3)
b1 = b0.item()              # 单tensor转标量


'''------------------------------------------------------------------------
Q. tensor的转置跟python不太一样，如何使用，如何避免not contiguous的问题？
1. python 用transpose(m,n,l)可以对3d进行转置，但tensor的transpose(a,b)只能转置2d
   要转置3d需要用permute()
2. 不连续问题解决办法：
    >data.contiguous()函数
    >data[...,[2,1,0]]切片运算
    >reshape()函数
'''
import torch
from numpy import random
# not contiguous的问题：来自pytorch的transpose/permute函数，用切片代替就不会有问题
# 解决方案1：a.contiguous()函数
# 解决方案2：用切片替代transpose/permute
# 解决方案3：用reshape替代view
a0 = torch.tensor(random.randint(1,10,size=(10,10,3)))
a1 = a0.permute(2,0,1)
a1.is_contiguous()  # permute后不连续
a1.view(10,5,6)     # 因为not contiguous报错
a1.contiguous().is_contiguous()
a1.contiguous().view(10,5,6)   # 解决(contiguous函数)
a1.reshape(10,5,6)     # 解决(reshape替代view)

a2 = a0[...,[2,0,1]]   # 解决(用切片代替transpose/permute)
a2.is_contiguous()  # 用切片后是连续的
a3 = a0[...,::-1]   # tensor还不支持负的step
a3.is_contiguous()  # 该操作还不能判断

b0 = torch.tensor(random.randint(1,10,size=(3,4)))
b1 = b0.transpose(1,0)
b1.is_contiguous()  # transpose后不连续




'''-------------------------------module----------------------------------
Q.在pytorch中model的本质是什么，有哪几种model
1. 核心概念：所有layer/model核心都是nn.module的继承，module基类包含了
2. 基类module的核心属性：
    _buffers(OrderedDict)，对应变量buffers(iterator)，存放所有
    _parameters(OrderedDict)，对应parameters(iterator)，存放所有
    _modules(OrderedDict)，对应modules(iterator)，存放所有子模型
3.module的核心运行逻辑
    >创建新的子模型时：conv1=nn.Conv2d(2,2,3), 则调用__setattr__更新_modules,_parameters,_buffers
    >
'''
# ------level1: module基类------
class Module(object):
    dump_patches = False
    _version = 1
    def __init__(self):
        continue
    def forward(self, *input):
        """这个方法用于主要的前向计算,是__call__方法实际调用的函数"""
        raise NotImplementedError
    def register_buffer(self, name, tensor):
        self._buffers[name] = tensor
        continue
    def register_parameter(self, name, param):
        self._parameters[name] = param
        continue
    def add_module(self, name, module):
        self._modules[name] = module
        continue
    def _apply(self, fn):
        """用于传入一个函数fn，实施到所有子module，同时实施到_parameters/_buffers"""
        for module in self.children():
            module._apply(fn)
        for param in self._parameters.values():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        return self
    def apply(self, fn):
        """用于传入一个函数fn，实施到所有子module上，该fn的参数输入必须是module"""
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self
    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))
    def cpu(self):
        return self._apply(lambda t: t.cpu())
    def type(self, dst_type):
        return self._apply(lambda t: t.type(dst_type))
    def float(self):
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)
    def double(self):
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)
    def half(self):
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)
    def to(self, *args, **kwargs):
        continue
    def register_backward_hook(self, hook):
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle
    def register_forward_pre_hook(self, hook):
        handle = hooks.RemovableHandle(self._forward_pre_hooks)
        self._forward_pre_hooks[handle.id] = hook
        return handle
    def register_forward_hook(self, hook):
        handle = hooks.RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle
    def _tracing_name(self, tracing_state):
        continue
    def _slow_forward(self, *input, **kwargs):
        continue
    def __call__(self, *input, **kwargs):
        continue
    def __setstate__(self, state):
        continue
    def __getattr__(self, name):
        continue
    def __setattr__(self, name, value):
        """定义了setattr方法，所以module.conv1=nn.conv2d()才能实现
        每次创建新的子模型，就会更新_parameters/"""
        self.register_parameter(name, value)
        modules[name] = value
        buffers[name] = value
        continue
    def __delattr__(self, name):
        continue
    def _register_state_dict_hook(self, hook):
        continue
    def _register_load_state_dict_pre_hook(self, hook):
        continue
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        continue
    def load_state_dict(self, state_dict, strict=True):
        continue
    def load(module, prefix=''):
        continue
    def parameters(self, recurse=True):
        """从named_parameters获得数据创建生成器"""
        for name, param in self.named_parameters(recurse=recurse):
            yield param
    def named_parameters(self, prefix='', recurse=True):
        """创建生成器"""
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem
    def buffers(self, recurse=True):
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf
    def named_buffers(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem
    def children(self):
        for name, module in self.named_children():
            yield module
    def named_children(self):
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module
    def modules(self):
        for name, module in self.named_modules():
            yield module
    def named_modules(self, memo=None, prefix=''):
        continue
    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
    def eval(self):
        return self.train(False)
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    def share_memory(self):
        return self._apply(lambda t: t.share_memory_())
    def _get_name(self):
        return self.__class__.__name__
    def __repr__(self):
        continue
    def __dir__(self):
        continue

# ------level2: layers继承Module的子类------
class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        continue
    def reset_parameters(self):
        """初始化module的参数"""
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
# ------level3: layers继承Module的子类------            
class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        continue
    def forward():
        """"""
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
# ------level4: layers函数------  
import torch.nn.functional as F
F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
    """对输入的多层图片数据进行卷积计算，返回tensor"""
# ------level5: Sequential类------ 
class Sequential(Module):
    """作为容器，有自己的_modules(OrderedDict)属性,存放所有子modules"""
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 支持解包list [OrderedDict]参数
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):       # 也支持解包list[module1, module2..]
                self.add_module(str(idx), module)
    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))
    def __setitem__(self, idx, module):
        """nn.Sequential()依靠__setitem__方法把子module加到父module的属性中去"""
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


'''------------------------------module-----------------------------------
Q. 如何创建module容器？
1. 可以用nn.Sequential(), nn.Sequential(), 前者传入的是list解包后的元素，后者传入的是list解包后的OrderedDict()
    后者可以方便增加名称       - 有实现forward()函数
2. 可以用nn.Modulelist(list) - 但forward()需要自己分层写
3. 可以用nn.ModuleDict(dict) - 但forward()需要自己分层写
'''
# 方式1: 直接输入每一层进sequential
model1 = nn.Sequential(nn.Conv2d(2,2,3),
                      nn.ReLU())
# 方式2: 先list，再解包
layers = [nn.Conv2d(1,2,3),nn.ReLU()]
model2 = nn.Sequential(*layers)
print(model2)
# 方式3: OrderedDict
model3 = nn.Sequential(OrderedDict([('conv1',nn.Conv2d(1,2,3)),
                                   ('re1', nn.ReLU())]))
print(model3)
# 方式4: OrderedDict的简写
model4 = nn.Sequential(OrderedDict(conv1=nn.Conv2d(1,2,3),
                                  re1=nn.ReLU()))
print(model4)
# 方式5: ModuleList
model5 = nn.ModuleList([])

# 方式6: ModuleDict
model6 = nn.ModuleDict(dict)


'''-----------------------------------------------------------------------
Q.如何便捷获取module的属性？
1. 通过module的3大字典属性_modules, _parameters, _buffers
2. 通过生成器方法modules(), parameters(), buffers(), children() - 只提供值
3. 通过生成器方法named_modules(), named_parameters(), name_buffers(), named_children() - 提供(名称,数值)
'''
# 单层模型
import torch.nn as nn
l1 = nn.Linear(2, 2)
l1._modules.keys()
l1._parameters.keys()
l1._buffers.keys()

# 多层模型
model = nn.Sequential(OrderedDict(conv1 = nn.Conv2d(2,2,3),
                                  relu1 = nn.ReLU()))
model._modules.keys()
for child in model.children():
    print(child)
for name, param in l1.named_parameters():
    print(name,': ',param)





'''-------------------------------module----------------------------------
Q.在pytorch中mmodule参数初始化方法有哪些，有什么区别？
1. 包含：
2. 
'''
init_weight = ???
net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weight)




