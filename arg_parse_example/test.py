#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 22:24:07 2019

@author: ubuntu
"""
import argparse
"""
运行方式
python ./test.py -v 13 --name leo 24 175   
python ./test.py 24 180 -v 13 --name jack

1. 位置参数必须输入，所以没有required选项，位置参数只要按顺序输入即可，放在前面后边都行
2. 关键字参数的required选项，用于定义是否必须输入
3. -v代表关键字参数，可以不输入
4. choice代表可选参数列表
5. default代表参数默认值
6. dest='a'代表可以通过args.a去访问
7. type用来定义输出参数类型，默认都是string，如果自定义type，就会自动做类型转换


实例2：运行mmcv的dist.sh脚本
python -m torch.distributed.launch --nproc_per_node 2 train_cifar10.py --launcher pytorch 就能启动分布式计算
1. torch.distributed.launch是pytorch用于启动分布式运算的命令, 该命令有


"""

def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('age', default='20')            # 位置参数，必须输入(所以也就没有requried这个参数)，按顺序即可，可以放前面，也可以放后边
    parser.add_argument('-v', required=True, type=int)  # -v表示关键字参数，输入参数作为变量了，变量名v
    parser.add_argument('--name', required=True)        # --name表示关键字参数，输入参数作为变量了，变量名name
    parser.add_argument('height')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('v = ', args.v + 20)
    print('name = ', args.name)
    print('age = ', args.age)
    print('height = ', args.height)