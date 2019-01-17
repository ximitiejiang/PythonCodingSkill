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

'''
Q. 对tensor的求和？
'''


# %%
'''-------------------------------module----------------------------------
Q.在pytorch中model的本质是什么，有哪几种model
1. 核心概念：所有layer/model核心都是nn.module的继承，module基类包含了
2. 基类module的核心属性：
    model._buffers，为OrderedDict变量，以下为相关函数：
        对应model.buffers(), 为iterator，存放所有buffers
        对应model.named_buffers(): 为iterator，存放带名称buffers
    model._parameters，为(OrderedDict)，以下为相关函数：
        对应model.parameters(): 为iterator，存放所有模型参数
        对应model.named_parameters(): 为iterator，存放带名称模型参数
    model._modules，为(OrderedDict)，以下为相关函数：
        对应model.named_modules，为(iterator)，是返回了所有有名称的module，包含了主模型/子模型/层
        对应model.modules，为(iterator)，从named_modules得到生成器
        还有model.named_chilren()，为iterator，是返回了子模型module
        还有model.children()，从named_children得到生成器

3. 基类module的重要方法：
    >model.add_module(name, new_model)用来添加子模型，等小于添加一个module的属性，比如model.sub_module_name = new_module
        所以self.add_module(name, new_model)等价于self.name = new_model
        可用来添加子模型
    >model.apply(fn)用来对每个子module实施fn,
        可用来init_weight
    >model.cuda(device=fn)用来调用_apply()把所有模型/子模型的参数都实施该fn(param.data)，
        可用来把模型参数传入device
    >model.load_state_dict() 用来加载已有模型的所有参数到本模型
        可用来导入预训练参数
    >model.train() 用于把module以及子模型的sefl.training标志位设置为True
        可用来实施training的指示
    >model.zero_grad() 用于把self.parameters里边所有参数的grad都设置为0
        可用来初始化梯度grad

3.module的核心运行逻辑
    >创建新的子模型时：conv1=nn.Conv2d(2,2,3), 则调用__setattr__更新_modules,_parameters,_buffers
    >
'''
# ------level1: module基类------
class Module(object):
    dump_patches = False
    _version = 1
    def __init__(self):
        self._parameters = OrderedDict() # 存放每个层的参数
        self._buffers = OrderedDict()    # 存放中间计算变量
        self._modules = OrderedDict()    # 存放子模型
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
        """这是model在运行的核心过程："""
        for hook in self._forward_pre_hooks.values():  # 先运行_forward_pre_hooks里边的hook
            hook(self,input)
        if torch._C._get_tracing_state():  # 这个应该是调试用的
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(*input, **kwargs)   # 计算模型输出
        for hook in self._forward_hooks.values():     # 再运行_forward_hooks里边的hook
            hook_result = hook()
        if len(self._backward_hooks)>0:               # 再运行_backward_hooks
            var = result                              
            while not isinstance(var, torch.Tensor):  
                if isinstance(var, dict):
                    var= next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)  # 把backward hook函数先绑定一个self形参
                    functools.update_wrapper(wrapper, hook)  # 更新hook的相关原始属性给wrapper
                    grad_fn.register_hook(wrapper)
        return result
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
# ------level5: Sequential类/ModuleList类/ModuleDict类------ 
class Sequential(Module):
    """作为modules子类，额外实现setitem/getitem作为切片手段,实现类似list的操作
    但跟list不同的是，创建时(init)接收的形参是解包后的元素或者OrderedDict，显得更像个初等函数"""
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 支持解包list [OrderedDict]参数，即直接丢OrderedDict进去
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):       # 也支持解包list[module1, module2..]，即直接丢解包后的module进去
                self.add_module(str(idx), module)
    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))
    def __getitem__(self, idx):
        continue
    def __setitem__(self, idx, module):
        """nn.Sequential()依靠__setitem__方法把子module加到父module的属性中去"""
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)
    def __delitem__(self, idx):
        continue
    def __len__(self):
        return len(self._modules)
    def __dir__(self):
        continue
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input
    
class ModuleList(Module): 
    """"借用module类的基本方法模拟出list的效果 (实现extend/append/insert)"""
    def __init__(self, modules):
        if modules is not None:
            self += modules    # 使用自定义iadd重载运算符
    def _get_abs_string_index(self, idx):
        continue
    def __getitem__(self, idx):
        continue
    def __setitem__(self,idx,module):
        continue
    def __delitem__(self, idx):
        continue
    def __len__(self):
        return len(self._modules)
    def __iter__(self):
        return iter(self._modules.values())
    def __iadd__(self):
        return self.extend(modules)
    def __dir__(self):
        continue
    def insert(self):
        continue
    def append(self,module):
        self.add_module(str(len(self)),module)
    def extend(self, modules):
        continue
    
class ModuleDict(Module):
    """"借用module类的基本方法模拟出dict的效果 (实现items/values/keys/update)"""
    def __init__(self, modules):
        if modules is not None:
            self.update(modules)
    def __getitem__(self, key):
        return self._modules[key]
    def __setitem__(self,idx,module):
        self.add_module(key,module)
    def __delitem__(self, idx):
        del self._modules[key]
    def __len__(self):
        return len(self._modules)
    def __iter__(self):
        return iter(self._modules)
    def clear(self):
        continue
    def pop(self,key):
        continue
    def keys(self, modules):
        return self._modules.keys()
    def items(self):
        return self._modules.items()
    def values(self):
        return self._modules.values()
    def update(self):
        continue
# %%
'''------------------------------module-----------------------------------
Q. 如何创建module容器, 以及组合module容器？
1. 可以用nn.Sequential(), nn.Sequential(), 前者传入的是list解包后的元素，后者传入的是list解包后的OrderedDict()
    后者可以方便增加名称       - 有实现forward()函数
2. 可以用nn.Modulelist(list) - 但forward()需要自己分层写
3. 可以用nn.ModuleDict(dict) - 但forward()需要自己分层写
(3个容器区别：Sequential是一个完整的带forward的module子类，可直接作为children module。而其他2中ModuleList/ModuleDict适合
先创建类，实现forward方法，然后在加入到一个主module中作为children module。)

4. 组合module容器
    >可以借用module的方法model.add_module(): 组合后的module作为子模型被加入_modules的字典中作为child_module
'''
import torch.nn as nn
from collections import OrderedDict
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
# 方式5: ModuleList, 配合list的所有方法(append/extend/insert)
layers = [nn.Conv2d(2,2,3) for i in range(10)]
model5 = nn.ModuleList(lst)
print(model5)
# 方式6: ModuleDict，配合dict的所有方法(pop/keys()/values()/update)
layers = dict(conv1 = nn.Conv2d(2,4,3),
              conv2 = nn.Conv2d(4,4,3),
              conv3 = nn.Conv2d(4,2,3))
model6 = nn.ModuleDict(layers)
print(model6)

# 基于ModuleList/ModuleDict需要额外实现forward
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ms = nn.ModuleList([nn.Conv2d(2,2,3) for i in range(5)])
    def forward(x):
        for i, m in enumerate(ms):
            if i // 2==0:
                x = m(x)
            else:
                x = nn.ReLU(m(x))
        return x

# 添加子模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.level1 = nn.Sequential(OrderedDict(   # 添加子模型的方式1：直接设置属性
                conv1 = nn.Conv2d(3,64,3),
                bn1 = nn.BatchNorm2d(64),
                relu1 = nn.ReLU()))
        
        self.add_module('level2', nn.Sequential(OrderedDict(   # 添加子模型的方式2：add_module()函数，等效于添加属性
                conv1 = nn.Conv2d(64,64,3),
                bn1 = nn.BatchNorm2d(64),
                relu1 = nn.ReLU())))
        
    def forward(self,x):
        x = self.level1(x)
        x = self.level2(x)
        return x
model = Net()

# 以下为了验证2种添加子模型的方式是完全等价，可以查看_modules/named_modules()/...
print(model)
print(model._modules.keys())
names = []
for name, module in model.named_modules():
    names.append(name)
print('total name len:{}'.format(len(names)))  # 输出主模型/子模型/层模型，只有主模型没有名字
print(names)



'''-----------------------------------------------------------------------
Q.如何便捷获取module的属性？
1. 通过module的3大字典属性_modules, _parameters, _buffers
2. 通过生成器方法modules(), parameters(), buffers(), children() - 只提供值
3. 通过生成器方法named_modules(), named_parameters(), name_buffers(), named_children() - 提供(名称,数值)
方法3使用最方便(因为里边包含了name/param比较全)，而named_children()比named_modules()更方便，因为named_modules()里边模型太完整不方便调用
'''
# 单层模型
import torch.nn as nn
l1 = nn.Linear(2, 2)
l1._modules.keys()
l1._parameters.keys()
l1._buffers.keys()
list(l1.named_modules())
list(l1.named_children())  # 单层没有named_children()

# 多层模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.level1 = nn.Sequential(OrderedDict(
                conv1 = nn.Conv2d(3,64,3),
                bn1 = nn.BatchNorm2d(64),
                relu1 = nn.ReLU()))
        self.level2 = nn.Conv2d(64,3,3)
    def forward(self,x):
        x = self.level1(x)
        x = self.level2(x)
        return x
model = Net()

model._modules.keys()                          # _modules里边保存的是子模型(类似named_children,不过一个是dict一个是iterator)

for module in model.named_modules():           # named_modules()输出所有主模型/子模型/层,其中主模型没名字，其他都预设用属性的名字
    print(module)
for name, module in model.named_modules():     # named_modules()每一个输出元素是tuple(name, module)
    print(name)    

for child in model.named_children():            # named_children()只输出子模型 
    print(child)
for name, module in model.named_children():     # named_children()每一个输出元素是tuple(name, module)
    print(name)
    
for name, param in model.named_parameters():    # named_parameters()每一个输出元素是tuple(name, parameter)
    print(name)


'''-------------------------------module----------------------------------
Q.在pytorch中几个基础模型的创建方式
1. vgg
2. resnet
'''
# ---------vgg-----------------
import torch.nn as nn
class VGG(nn.Module):
    """vgg模型的实现："""
    arch_settings = {
            11: (1, 1, 2, 2, 2),
            13: (2, 2, 2, 2, 2),
            16: (2, 2, 3, 3, 3),
            19: (2, 2, 4, 4, 4)}  # 记住2,2,4,4,4,这个vgg19是最常用
    def make_vgg_layer(self, num_blocks):     # 记住每个block结构(conv3x3+bn+relu)*n + maxpool
        layer = []
        for i in range(num_blocks):
            layer.append(nn.Conv2d(inplane, outplane, 3, padding=padding, dilation=dilation))
            if bn:
                layer.append(nn.BatchNorm2d())
            layer.append(nn.ReLU())
        layer.append(nn.MaxPool2d())
        return layer
    
    def __init__(self,depth, 
                 with_bn=True,
                 dilations=[1,1,1,1,1],
                 out_indics=[],
                 with_last_pool=false):
        self.out_indics = out_indics
        blocks = arch_settings.get(depth)
        layers = []
        for block in blocks:
            layers.extend(make_vgg_layer(block))
        if not with_last_pool:
            layers.pop(-1)
        vgg_layer = nn.Sequential(*layers)
        
        
    def forward(self,x):
        out = []
        features = model(x)
        for layer in self.out_indics:
            out.append(features[layer])
        
# ----------resnet----------------
class BasicBlock:
    """可以先用moduleList实现子模块,，但需要实现forward，然后作为children module加入主模型"""
    def __init__(self):
        pass
    def forward(self):
        pass

class BottleNeck():
    def __init__(self):
        pass
    def forward(self):
        pass
    
class Resnet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))}
    def make_resnet_block(self):
        pass
    def __init__(self):
        pass
    def __forward__(self):
        pass



'''-------------------------------module----------------------------------
Q.在pytorch中conv2d的基本计算过程以及各个参数的作用？
'''
m = nn.Conv2d(3,3,3)
from numpy import random
random.seed(11)
a = random.uniform(0,1, size=(3,5,5))
input = torch.tensor(a)
output = m(input)
print(output)



'''
Q. 对于下采样时conv2d/maxpool的设置区别？
'''
# 作为下采样功能：
# 可以用conv2d/maxpool，对应s=2, 但两者由于计算w/h方式不同所以kernel size一般不同
# conv2d的k-size取3, (w-3+2)/2+1为整数，maxpool的k-size取2, (w-2)/2+1为整数
conv1 = nn.Conv2d()
maxpool1 = nn.MaxPool2d()


    
'''-------------------------------module----------------------------------
Q.在pytorch中mmodule参数初始化方法有哪些，有什么区别？
1. 包含：
2. 
'''
init_weight = ???
net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weight)


'''-------------------------------module----------------------------------
Q.在pytorch中的data paralle模块如何实施
'''
# 熟悉Data Parallel类
# 原始pytorch执行流程：data parallel -> forward -> scatter -> scatter_kwargs -> scatter -> scatter_mape -> Scatter.apply()
# 新的mmdetection流程：MMdataParallel -> forward -> scatter(*) -> scatter_kwargs(*) - > scatter(*) -> scatter_map(*) -> Scatter.forward(*)
class DataParallel(nn.Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        continue
    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)  # 调用self.scatter
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        
        # 复制模型：replicate()函数，包含复制主模型(__dict__/_parameters/_buffers/_modules)，复制子模型(_modules/_parameters/_buffers)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])  
        # 并行计算: 首先需要线程锁threading.Lock()，同时创建多线程，
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)
    def scatter(self,inputs, kwargs, device_ids):
        """这是pytorch原有的scatter()"""
        return scatter_kwargs(inputs, kwargs, device_ids, dim =self.dim)
    
    def scatter():
        """这个是子类MMDataParallel的scatter()重写函数，调用的也是重写的scatter_kwargs
        如果是MMDataParallel类(继承DataParallel)则用这个scatter覆盖了父类scatter()
        """
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

def scatter_kwargs():
    """这个也是重写的，但跟pytorch的原函数一样"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

def scatter(inputs, target_gpus, dim=0):
    """这个也是重写的，增加对DataContainer的支持"""
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return OrigScatter.apply(target_gpus, None, dim, obj)  # OrigScatter是pytorch原有Scatter类
        if isinstance(obj, DataContainer):    # 新增对DataContainer的数据对象的支持
            if obj.cpu_only:
                return obj.data
            else:
                return Scatter.forward(target_gpus, obj.data)  # 重写Scatter()类，并加了一个forward()方法
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]
    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None
        
class Scatter(Function): # 原版scatter继承自Function基类
    """这是pytorch的原版Scatter类, 没有apply方法，Scatter.apply()是继承自
    Function父类的父类_C._FunctionBase，也就是C语言写的
    """
    @staticmethod
    def forward():
        continue
    @staticmethod
    def backward():
        continue
    
class Scatter(object):  # 重写Scatter跟原版Scatter没有关系，不是继承
    """这是重写的Scatter类，仅包含一个forward()函数，处理DataContainer数据类型"""
    @staticmethod
    def forward(target_gpus, input):
        input_device = get_input_device(input)
        streams = None
        if input_device == -1:
            # Perform CPU to GPU copies in a background stream
            streams = [_get_stream(device) for device in target_gpus]

        outputs = scatter(input, target_gpus, streams)
        # Synchronize with the copy stream
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs)

# step1: 对model进行wrap成DataParallel(model)
model = DataParallel(model)
# step2: 对input进行scatter()
inputs = Scatter.forward()
# step3: 对model进行复制replica()
def replica():
    continue
# step4: 对多model多参数进行多线程并行计算
def parallel_apply():
    continue

