#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:00:21 2019

@author: ubuntu
"""

# %%
"""在linux系统里如何使用sh文件和相关命令编程进行自动运行？
1. sh脚本和makefile的区别：
    sh和makefile都是脚本语言，但makefile似乎主要专注在对c/cpp进行编译上；而sh脚本是一个更广泛使用的东西，他能调用makefile
2. sh脚本的基本语法：
    >只要在命令行能够运行的指令都可以直接写在sh脚本文件中
    >需要创建一个.sh文件，然后在命令行运行$ sh filename.sh
    >等号左右不能有空格，否则报错
    >变量赋值类似python，不过只有2种数据类型，一种整数，一种字符串
    >变量计算或者输出，都必须带$，计算还必须包含2组括号，且计算结果只能取整(下取整floor)
"""
# 赋值
a='hello world'
b=100
c=99

# 数据计算: 计算带$
d=$((b+c))
e=$((c/b))

# 输出打印: 变量带$
echo 'a='$a
echo 'd='$d
echo 'e='$e
echo 'this is the ${b}th num'

# 条件判断
if [ -d "build" ]; then
    rm -r build
fi

# 进入某个目录
cd ./ssd                # 进入当前目录下的ssd文件夹
cd ../data              # 进入上一级目录下的data文件夹
cd ..                   # 退出当前目录

# 创建文件夹
mkdir name

# 复制文件
cp src_file dst_file

# 移动文件
mv src_file dst_file

# 删除文件夹下文件
rm a.py         # 只能用来删除文件，不能删除文件夹
rm -r build     # 删除当前文件夹下的build文件夹
rm -rf build    # 强制删除该文件夹内的文件和该文件夹(r代表递归删除，f代表强制)
                # 这个命令很凶险，删除时不会提示确认，也不会放入回收站也找不回文件，慎用！
shopt -s extglob      # 开启内置筛选选项：从而可以使用!筛选符号
rm !(*.zip)           # 只保留zip文件
rm !(*.zip|*.iso)     # 只保留zip和iso文件

# 获得python版本
PYTHON=${PYTHON:-"python"}

# 执行python文件
$PYTHON setup.py build_ext --inplace

# 执行make文件
make clean               # 执行该目录下的make文件中的clean段 
make PYTHON=${PYTHON}    # 执行该目录下的make文件：此时会执行makefile中all段

# -------------其他一些辅助的没有操作的linux系统命令-------------
# 显示当前路径
pwd
# 显示当前路径那个下的文件夹和文件
ls
# 安装一个可以展示目录树的工具
sudo apt install tree
tree ssd       # 同时显示文件夹和文件
tree -L 1 ssd  # 只展示L1层级的文件夹和文件

ssd
├── compile.sh
├── config
├── data
├── dataset
├── model
├── README.md
├── requirements.py
├── test
├── TEST_ssd_coco.py
├── TEST_ssd_img.py
├── TRAIN_ssd_coco.py
├── TRAIN_ssd_voc.py
├── utils
└── weights



# %%
"""如何使用简单的makefile，用来对build
0. makefile的功能：用于描述整个c++工程的编译/链接的规则，就包括那些源文件需要编译，如何编译，需要哪些库文件，如何产生可执行文件。
虽然makefile编写事无巨细都要定义，但只要定义完成后整个工程的自动化编译就只需要一句make，很方便。
所谓编译，就是把源文件编译成中间文件，linux下中间文件是.out文件，windows下是.obj文件，这就是compile
所谓链接，就是把大量编译文件.o合成一个执行文件，这就是link

1. 基本makefile的写法：
    target ... : prerequisites ...
        command
        ...
    其中target可以是编译的.out文件，也可以是链接的可执行文件
    其中prerequisites就是生成target所需要的文件
    其中command就是make需要执行的shell命令
    makefile执行过程：他会比较target与prerequisites的文件修改日期，如果targets较早，则更新，如果targets不存在，则执行command
    
1. 常见处理方法如下：相比之下我喜欢用sh文件直接调用setup.py这样省去了makefile
    compile.sh 调用setup.py,  这种方式好处是省略了写makefile
    compile.sh 调用makefile，然后makefile再调用setup.py，这种方式好处是可以单独运行make，且不用输入文件名
"""
# 获得python版本
PYTHON=${PYTHON:-python}
# 创建目标
all:       # 创建终极目标all：通常用来编译所有目标
    $(PYTHON) setup.py build_ext --inplace   # 调用setup.py文件进行编译
clean:     # 创建伪目标clean：目的是删除所有被make创建的多余文件
    rm -r *.so   # 删除所有so文件


# %%
"""在linux系统用指令创建软链接？
"""
$ cd ssd_detector
$ mkdir data
$ ln -s /home/ubuntu/MyDatasets/coco
$ ln -s /home/ubuntu/MyDatasets/voc/VOCdevkit



# %%
"""在linux系统中如何设计断点续传
"""
$ wget -c https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth



# %%
"""如何在ipython调试软件中设置让图片单独显示而不是嵌入在命令行？
参考：https://blog.csdn.net/mozai147/article/details/79850065
"""
#命令行输入：代表嵌入式显示
%matplotlib inline
#命令行输入：代表独立显示
%matplotlib qt5
# spyder中设置：inline代表嵌入显示，qt5代表独立显示
Tools > Preferences > IPython Console > Graphics > Graphics backend


# %%
"""如何向开源仓库提交PR
"""
git remote -v  # 查看当前目录跟什么仓库建立了连接: 一开始只会跟自己fork的仓库建立连接

git remote add upstream https://github.com/qijiezhao/M2Det  # 跟原始仓库建立fetch/push

git remote -v  # 此时就可以看到跟2个仓库分别建立了fetch/push的连接

git checkout -b mydev  # 创建一个分支，叫做mydev，并自动切换到这个分支
# 然后就可以修改代码了
git add .                # 增加全部更改

git commit -m 'update xxx'   # 提交

git push origin mydev:mydev  # 从我本地mydev分支提交到我的远程mydev分支
# 最后在我的远程仓库点击send pull request


../data/coco/annotations/instances_train2017.json
../data/coco/annotations/instances_train2017.json


# %%
"""基本的markdown语法？
参考：https://www.jianshu.com/p/191d1e21f7ed
"""
# 一级标题
## 二级标题
### 三级标题

**加粗**
*斜体*
***斜体加粗***
~~删除线~~

>引用(产生一条竖线缩进)
>>二级引用(产生两条竖线缩进)

---(水平分割线)
***(水平分割线)

![图片描述](图片地址 "图片title")  # 其中图片描述显示在图片下方，图片title在鼠标移到图片上显示，title可加可不加

[超链接名称](超链接地址 "超链接title")  # 其中超链接名称就是显示的描述，title也是可加可不加

+ 列表内容(产生一个列表小圆点，也可用-代替+)
   + 列表内容(前面加3个空格产生二级列表嵌套，也可用-代替+)

表头|表头|表头
---|:--:|---:   # 这一行是分割表头和内容，中间的-号有一个就行，但为了对齐可以增加n个
内容|内容|内容  # 文字对齐通过:控制，两边都加:则中间对齐，左边加:则左对齐，右边加:则右对齐

`单行代码`

```
多行代码
```

- [] 未处理事项
- [X] 已处理事项
# 基本上就这些了