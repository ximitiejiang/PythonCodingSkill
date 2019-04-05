#!/bin/sh
# 这句是告诉系统需要使用sh解析器，也有写成#!/bin/bash也就是使用bash解析器


# --------------------sh文件和makefile文件的应用区别？--------------------
# 我当前的理解是makefile也是脚本，但主要专注在对c/cpp进行编译上。
# 而sh脚本是一个更广泛使用的东西，他能调用makefile，


# --------------------变量赋值，等号两边不能有空格--------------------
a='hello world'
b=100
c=99

# --------------------数据格式和计算--------------------
# 加减乘除运算需要两层小括号，且不能有空格：$(())
# 数据格式只有整数和字符串，所以相除得到取整(且为下取整)
d=$((b+c))
e=$((c/b))

# --------------------变量打印需要带$前缀--------------------
echo 'a='$a
echo 'd='$d
echo 'e='$e
echo 'this is the ${b}th num'

# --------------------进入某个目录--------------------
cd ./simple_ssd_pytorch

# --------------------获得某个软件的版本--------------------
#这里是获得python的版本
PYTHON=${PYTHON:-"python"}

# --------------------执行某个py文件--------------------
# 由于是执行python的文件，所以需要增加前缀$PYTHON
$PYTHON setup.py build_ext --inplace

# --------------------判断某个目录下是否有文件夹--------------------
# if ; then  fi
# 注意if尾部需要有分号，fi代表end if
# [-d name]代表如果name存在且为目录
# [-a name]代表如果
if [ -d "build" ]; then
    rm -r build
fi
    
# --------------------删除文件夹下的文件-------------------- 
# rm -r指删除全部目录和子目录
rm -r build

# --------------------调用makefile中的目标--------------------
make clean               # 执行该目录下的make文件中的clean段 
make PYTHON=${PYTHON}    # 执行该目录下的make文件：此时会执行makefile中all段

# --------------------一个sh脚本实例，实现调用makefile--------------------
PYTHON=${PYTHON:-"python"}                  # 获得python版本
echo "Building roi pool op..."              # 打印
cd ../roi_pool                              # 进入目录
if [ -d "build" ]; then                     # 判断如果存在某文件夹就删除
    rm -r build
fi
$PYTHON setup.py build_ext --inplace        # 调用py文件

echo "Building nms op..."
cd ../nms
make clean                                  # 调用makefile文件的clean子段
make PYTHON=${PYTHON}                       # 调用makefile文件的all字段(默认all子段)

