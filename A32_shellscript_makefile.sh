#!/bin/sh
# 这句话不能省略：是告诉系统需要使用sh解析器，也有写成#!/bin/bash也就是使用bash解析器

# 参考http://www.runoob.com/linux/linux-shell.html


# --------------------export命令--------------------
export NGPUS=2   # 在当前登录下添加一个环境变量NGPUS=4
                 # 只要该terminal终端关闭或者换一个终端，就相当于该次登录退出，该变量即消失

gedit ~/.bashrc  # 如果要永久添加某个环境变量，就要在bashrc文件中添加
export NGPUS=2
source ~/.bashrc

export -p     # 显示当前所有环境变量
echo $NGPUS   # 显示某个环境变量


# --------------------export PYTHONPATH 操作--------------------
# 方法1只针对当前终端：如果只是export，则只是针对当前终端，一旦当前终端关闭或在另一个终端，则路径无效
# 方法2只针对当前用户：如果先打开.bashrc文件，gedit ~/.bashrc，然后再export PATH=...$PATH，则当前用户每次登录都有效
# 方法3可适用所有用户：可先打开sudo gedit /etc/profile，然后再export PATH=...$PATH
# 其中，方法1是立即生效，方法2/3都需要保存以后才生效
# 大部分github里边的软件，都采用安装egg的方式加到PYTHONPATH中，少部分用方法1针对当前终端，基本没有添加到bashrc或profile文件中去的情况
# PATH是指系统环境变量(可echo $PATH查看)，而PYTHONPATH是PYTHON的环境变量，等效于sys.path
export PYTHONPATH=/home/ubuntu/suliang_git/pysot:$PYTHONPATH    # 命令行添加方式

import sys
sys.path.append('/home/ubuntu/suliang_git/pysot')      # python脚本添加方式
 

# --------------------Terminal的快捷操作--------------------
cd suli + tab  # 用tab可以对当前目录下的子目录地址自动补全
ctrl + r      # 可以reverse search搜索历史命令，比如按下ctrl+r之后输入wget就可以找到之前下载的命令地址


# --------------------注释的3种方法--------------------
# echo "aaa"           # 注释方式1：用#对单行内容注释   

ex(){                  # 注释方式2：用函数对多行内容注释
echo "this is a test"
echo "this is a test too"
}

if false; then         # 注释方式3：用条件语句注释
echo "this is a test"
echo "this is a test too"
fi


:<<!                  # 注释方式4：用4个字符:<<!开头，用!结尾来对多行内容注释
echo "this is a test"
echo "this is a test too"
!

# --------------------运行shell脚本文件--------------------
chmod +x test.sh     #chmod代表change mode也就是变更操作权限，+代表增加权限，x代表可执行，+x代表增加可执行权限
./test.sh     # 执行脚本：必须写成./test.sh让系统在当前目录下寻找文件，而不是test.sh否则系统会去PATH里边寻找test.sh

sh test.sh    # 运行解释器sh, 而sh文件作为解释器参数.这种方式不需要增加执行权限，是更方便的方式。前一种运行方式每次都要增加运行权限，比较麻烦

# --------------------向shell脚本文件传递参数--------------------
# 假定如下是test.sh脚本文件内容：
echo "document name: $0"     # $0表示文件名
echo "this is param1: $1"    # $1～$n代表第几个参数
echo "this is param2: $2"
echo "this is param3: $3"    
echo "total params: $#"      # $#代表所有参数个数
echo "show param as str: $*" # $*代表所有参数，以字符串形式

# 执行test.sh脚本，并传入参数
./test.sh i love you

# --------------------shell脚本文件形参的解包方式--------------------
"$*"  # 代表的是把所有形参合成一个字符串
"$@"  # 代表的是把所有形参分成n个字符串

# --------------------变量赋值，等号两边不能有空格--------------------
a="hello world"   # 注意等号左右不要有空格，注意字符串最好用双引号""，虽然单引号有时候也能操作，但组合多个字符串时单引号就会失效。
b = 'hello world'  # 单引号字符串中间不能有变量，不能有转义字符
b=100             # 变量名只能英文/数字/下划线，不能数字开头
c=99
readonly aaa      # 创建只读变量
aaa=10

# --------------------变量删除--------------------
unset a    # 删除变量a，但不能删除只读变量

# --------------------变量的使用：带$前缀--------------------
a=10
echo ${a}       # 花括号可加可不加都行，但建议把所有变量都加上花括号，这是个好的编程习惯
name="BBB"
echo "My name is AAA.${name}.CCC"  # 花括号用来区分变量边界
echo $a

# --------------------变量打印需要带$前缀--------------------
echo 'a='$a
echo 'd='$d
echo 'e='$e
echo 'this is the ${b}th num'

# --------------------变量打印的2种方式--------------------
a = "abc"

echo $a

echo "hello world"
printf "hello world"


printf "%-10s %4.2f\n" name age    # \n代表转义字符换行。 %-10s代表10个字符左对齐
printf "%-10s %4.2f\n" leo 32.145  # %-4.2f代表保留2位小数左对齐
printf "%-10s %4.2f\n" eason 8.1



# --------------------字符串的操作--------------------
st="abcdefghijklmnopqrstuvwxyz"
echo ${#st}       # 获得字符串长度：用#号
st2="i love $st"  # 拼接多个字符串：用双引号
echo $st $st2     # 拼接多个字符串

echo ${st:1:4}   #提取字符串的

# --------------------生成数组和数组操作--------------------
array=(1 2 3 4 5)    #小括号包含数值来建立数组，数值之间空格
array[8]=100         # 对某一下标的数组位置赋值，如果中间没有赋值过，则为空，比如此时array[5]~array[7]都为空


echo ${array[2]}     #提取第2个数值：下标从0开始，注意数组值的提取一定要有花括号
echo ${array[*]}     #提取所有元素
echo ${array[@]}     #提取所有元素
lenth=${#array[*]}   #获取数组长度


# --------------------生成序列--------------------
seq 10     # 从1到20的数组，不是从0开始
seq 2 20   # 从2到20的数组，2和20都包含

# --------------------运算符--------------------
a=10
b=8
val=`expr $a + $b`     #用`expr`来进行表达式运算，注意要用反引号，主要表达式与变量之间要空格
val2=`expr $a % $b`    # %代表取余
echo `expr $a \* $b`   # 相乘必须用\*而不能直接用*否则报错
echo `expr $a + $b`    # 可以嵌套在echo语句中,且不需要用单引号或者双引号，因为expr已经有引号了。

if [ $a == $b]         # 条件运算符==(等于)，!=(不等于)
then
    echo "a equal b!"
fi

if [ $a -lt $b]        # -lt/-le/-gt/-ge/-eq/-ne分别表示小于/小于等于/大于/大于等于/等于/不等于
then
    echo "a less than b"
else
    echo "a geat than b or a equal b"
fi

if [ $a -lt 5 -a $b -gt 8 ]   # -a/-o/! 分别表示and与/or或/非
then                          # &&/|| 分别表示逻辑and/or
    echo "a<5 and b>8"

# --------------------字符串检测运算符--------------------
str="abc"
if [ -z str]   # -z检测字符串是否长度为空，是则返回true, -z代表zero时返回true, -n代表not zero时返回true
then
    echo "str is empty"
fi

if [ $str ]    # 直接检测字符串是否空，如果不空则true
then
    echo "str is valid, not empty"
fi

# --------------------文件检测运算符--------------------
file="./ssd"
if [ -d file]  # 检测是否为文件夹, d代表dir

if [ -f file]  # 检测是否为文件，f代表file

if [ -x file]  # -x/-r/-w/-s分别代表可执行文件/可读文件/可写文件/空文件(文件大小为0)


# --------------------条件语句--------------------
a=1              # if 结尾要接一个fi, 记住fi就是反过来的if，也就表示结束。
if [a==1]        # 普通if then fi语句，分行写，不需要分割号
then
    echo "a equals 1"
fi

if [a==1]; then echo "yes!"; fi    # 单行条件语句，用分号隔开，用于在命令行输入
    
if [ $a == $b]
then
    echo "a equals b"
elif [ $a -gt $b]
then
    echo "a great than b"
fi

# --------------------循环语句--------------------
for i in `seq 2 20`:  # 用`seq 2 20` 表示一组数进行循环
do
    echo "i=${i}"   # 或者写成 echo "i=${i}" 或者是echo "i="${i}
done

for i in 1 2 3 4 5   # 直接写一组数进行循环
do 
    echo "${i}"
done

for s in "this is good way"  # 只能被一次性取出
do
    echo "${s}"
done    


num=0
while(($num<=5))   # 这段报错，还没调通
do
    echo "$num"
    let "num++"
done


# --------------------函数--------------------
demoFun(){
    echo "this is my first func"
}

demoFun


# --------------------进入某个目录--------------------
cd ./ssd                # 进入当前目录下的ssd文件夹
cd ../data              # 进入上一级目录下的data文件夹
cd ..                   # 退出当前目录

cd ~           # 进入用户的Home目录，一般每个用户有自己的home目录，可通过pwd打印出来，比如我的就是/home/ubuntu

vim ~/.vimrc   # 表示用vim 打开/home/ubuntu/下面这个.vimrc隐藏文件(这是在个人home目录下对vimi进行定制的一个配置文件) 


# --------------------创建文件夹--------------------
mkdir ./a/b             # 只能创建b文件夹，且a必须存在
mkdir -p ./a/b          # 其中-p代表如果路径中有文件夹不存在，就会自动创建，也就是能同时创建路径中多个嵌套文件夹，比如a,b都可以不存在

# --------------------打印当前文件夹路径--------------------
pwd

# --------------------显示当前文件夹下所有子文件夹和子文件--------------------
ls

# --------------------创建文件夹快捷方式--------------------
ln -s abc/bc/c    # -s 代表symlink

# --------------------创建文件方法1： 用vim filename--------------------
vim main.c       # 创建main.c文件
vim makefile     # 创建makefile文件

# --------------------创建文件方法2: cat命令一般用来创建文件，查看文件内容，文件合并，追加文件内容能-----------
# cat的本意是concatenate and print, 所以主要是用于合并和输出，输出到空文件就是创建了
# 参考：https://www.cnblogs.com/fabulousyoung/p/4079759.html
cat >main.c<<EOF    # 创建文件用>file_name指定文件名，并且要定义成对出现的文件结束标志位<<EOF 文件内容 EOF
EOF

cat >>main.c<<EOF   # 追加文件内容用>>file_name制定文件名，注意创建用单箭头但追加用双肩头
EOF

cat f1.txt  # 在终端查看文件内容

cat f1.txt f2.txt>f3.txt  # 合并2个文件成1个文件，并清空f3.txt后写入合并的内容
cat f1.txt f2.txt>>f3.txt # 合并2个文件成1个文件，不清空f3.txt而是在末尾追加写入

# --------------------复制文件--------------------
cp src_file dst_file

# --------------------移动文件--------------------
mv src_file dst_file

# --------------------下载文件--------------------
curl -LO http://images.cocodataset.org/zips/train2017.zip

# --------------------下载文件--------------------
$ wget -c https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth

# --------------------解压缩文件--------------------
unzip -qqjd ../images ../images/train2017.zip


# --------------------删除文件夹下的文件-------------------- 
# rm -r指删除全部目录和子目录
rm a.py         # 只能用来删除文件，不能删除文件夹
rm -r build     # 删除当前文件夹下的build文件夹
rm -rf build    # 强制删除该文件夹内的文件和该文件夹(r代表递归删除，f代表强制)
                # 这个命令很凶险，删除时不会提示确认，也不会放入回收站也找不回文件，慎用！
shopt -s extglob      # 开启内置筛选选项：从而可以使用!筛选符号
rm !(*.zip)           # 只保留zip文件
rm !(*.zip|*.iso)     # 只保留zip和iso文件


# --------------------计算运行时间--------------------
start=`date +%s`
end=`date +%s`
runtime=$((end-start))


# --------------------获得某个软件的版本--------------------
#这里是获得python的版本
PYTHON=${PYTHON:-"python"}

# --------------------执行某个py文件--------------------
# 由于是执行python的文件，所以需要增加前缀$PYTHON
$PYTHON setup.py build_ext --inplace

$python3 abc.py          # 这种运行方式叫直接运行，此时在sys.path中会增加该abc.py文件所在文件夹的目录
$python3 -m abc.py       # 这种运行方式叫导入模块，此时在sys.path中会增加''，这里''代表当前目录，也就是把输入命令时的目录加到sys.path

# 什么时候选择什么运行方式：参考https://www.cnblogs.com/xueweihan/p/5118222.html
# 如果abc.py文件中有一句import 其他文件夹文件，此时如果用python3 abc.py只会在该文件所在文件夹中搜索，会导致导入失败
# 而如果用python3 -m abc.py此时sys.path会有''代表该文件所在目录，导入才能成功?????


# --------------------判断某个目录下是否有指定文件夹--------------------
# if ; then  fi
# 注意if尾部需要有分号，fi代表end if
# [-d name]代表如果name存在且为目录
# [-a name]代表如果
if [ -d "build" ]; then
    rm -r build
fi
    



# --------------------在线安装一些系统软件--------------------
sudo apt-get install           # apt-get 代表advanced package tool是一个应用程序管理器， 其中install指令就是从网络获取软件并安装在本机


# --------------------打印当前文件目录树--------------------
sudo apt install tree
tree ssd       # 同时显示文件夹和文件
tree -L 1 ssd  # 只展示L1层级的文件夹和文件

#
#1. sh脚本和makefile的区别：
#    sh和makefile都是脚本语言，但makefile似乎主要专注在对c/cpp进行编译上；而sh脚本是一个更广泛使用的东西，他能调用makefile
#2. sh脚本的基本语法：
#    >只要在命令行能够运行的指令都可以直接写在sh脚本文件中
#    >需要创建一个.sh文件，然后在命令行运行$ sh filename.sh
#    >等号左右不能有空格，否则报错
#    >变量赋值类似python，不过只有2种数据类型，一种整数，一种字符串
#    >变量计算或者输出，都必须带$，计算还必须包含2组括号，且计算结果只能取整(下取整floor)


# makefile的功能：用于描述整个c++工程的编译/链接的规则，就包括那些源文件需要编译，如何编译，需要哪些库文件，如何产生可执行文件。
# 虽然makefile编写事无巨细都要定义，但只要定义完成后整个工程的自动化编译就只需要一句make，很方便。
# 所谓编译，就是把源文件编译成中间文件，linux下中间文件是.out文件，windows下是.obj文件，这就是compile
# 所谓链接，就是把大量编译文件.o合成一个执行文件，这就是link

#1. 基本makefile的写法：
#    all ... : prerequisites ...
#        command
#    targetA: 依赖1 依赖2    ...
#        command
#    其中target可以是编译的.out文件，也可以是链接的可执行文件
#    其中prerequisites就是生成target所需要的文件
#    其中command就是make需要执行的shell命令
#    makefile执行过程：他会比较target与prerequisites的文件修改日期，如果targets较早，则更新，如果targets不存在，则执行command
#    
#1. 常见处理方法如下：相比之下我喜欢用sh文件直接调用setup.py这样省去了makefile
# c++语言嵌入python中编译过程稍有差异
#   1. 如果是c++本身的编译，通常是写一个makefile作为编译工具，然后直接make即可
#   2. 如果是python调用c++，通常是写一个setup.py作为编译工具，然后用sh脚本调用运行这个setup文件即可
# 如下是一个setup.py文件用于编译

# --------------------make指令--------------------
make              # 执行当前路径下的makefile文件，且只能执行文件名为makefile/Makefile的文件
make -f filename  # 执行指定文件，且可以执行任何自定义脚本文件，比如make -f test.mk
make target1      # 执行makefile中的某一个target

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

# --------------------makefile语法跟shell语法有区别--------------------
object=program.o foo.o utils.o   # 定义一个变量
$ (object)                       # 引用一个变量
${object}                        # 引用一个变量

$@      # 自动化变量：代表规则中的目标文件名
$<      # 代表规则中地一个依赖文件名
$^      # 代表规则中所有以来文件列表，文件名用空格分割
$$      # 把$$转义成普通字符$
\$$@     # 把$@转义成普通字符串$@


# --------------------conda源的添加和删除--------------------
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

conda config --remove-key channels  # 移除

