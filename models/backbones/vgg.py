import logging

import torch.nn as nn

from models.backbone_support.weight_init import constant_init, normal_init, kaiming_init
from models.backbone_support.checkpoint import load_checkpoint


def conv3x3(in_planes, out_planes, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=dilation,
        dilation=dilation)


def make_vgg_layer(inplanes, planes, num_blocks, dilation=1, with_bn=False,
                   ceil_mode=False):
    """create vgg block: (conv3x3 + bn + relu) * n + maxpool
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(conv3x3(inplanes, planes, dilation))
        if with_bn:
            layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        inplanes = planes
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

    return layers


class VGG(nn.Module):
    """VGG backbone.

    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        with_bn (bool): Use BatchNorm or not.
        num_classes (int): number of classes for classification.
        num_stages (int): VGG stages, normally 5.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers as eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
    """

    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }

    def __init__(self,
                 depth,
                 with_bn=False,
                 num_classes=-1,
                 num_stages=5,
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3, 4),
                 frozen_stages=-1,
                 bn_eval=True,
                 bn_frozen=False,
                 ceil_mode=False,
                 with_last_pool=True):
        super(VGG, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for vgg'.format(depth))
        assert num_stages >= 1 and num_stages <= 5
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        assert len(dilations) == num_stages
        assert max(out_indices) <= num_stages

        self.num_classes = num_classes
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen

        self.inplanes = 3
        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []
        for i, num_blocks in enumerate(self.stage_blocks): #based on arch_setting
            num_modules = num_blocks * (2 + with_bn) + 1  #(conv+relu+bn)*blocks + maxpool
            end_idx = start_idx + num_modules
            dilation = dilations[i]
            planes = 64 * 2**i if i < 4 else 512
            vgg_layer = make_vgg_layer(
                self.inplanes,
                planes,
                num_blocks,
                dilation=dilation,
                with_bn=with_bn,
                ceil_mode=ceil_mode)
            vgg_layers.extend(vgg_layer)
            self.inplanes = planes
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx
        if not with_last_pool:
            vgg_layers.pop(-1)
        self.module_name = 'features'
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i, num_blocks in enumerate(self.stage_blocks):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        """重写nn.Module的train()方法，用于设置在eval模式下bn层设置和bn参数冻结模式
        以及在train模式下其他层的冻结参数设置
        """
        super(VGG, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        vgg_layers = getattr(self, self.module_name)
        if mode and self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                for j in range(*self.range_sub_modules[i]):
                    mod = vgg_layers[j]
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False


if __name__ == '__main__':
    vgg16 = VGG(depth=16,  # 测试ssdvgg所用的vgg16模型
                 with_bn=True,
                 num_classes=20,
                 num_stages=5,
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3, 4),
                 frozen_stages=-1,
                 bn_eval=False,
                 bn_frozen=False,
                 ceil_mode=False,
                 with_last_pool=True)    
#   由于是定制的vgg16, 没有相应的预训练参数可用，且不能使用原有的pytorchmodel参数
    sub=[[],[]]
    for i,(_, module) in enumerate(vgg16.named_children()):
        for name,_ in module.named_children():
            sub[i].append(name)
    print(len(sub[0]), len(sub[1]))   # 2个children module，分别包含53 + 7个层
    

    
