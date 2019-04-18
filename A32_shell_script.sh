#!/bin/sh
# 这句话不能省略：是告诉系统需要使用sh解析器，也有写成#!/bin/bash也就是使用bash解析器

# 参考http://www.runoob.com/linux/linux-shell.html

# --------------------sh文件和makefile文件的应用区别？--------------------
# 我当前的理解是makefile也是脚本，但主要专注在对c/cpp进行编译上。
# 而sh脚本是一个更广泛使用的东西，他能调用makefile，


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

# --------------------创建文件夹--------------------
mkdir ./a/b             # 只能创建b文件夹，且a必须存在
mkdir -p ./a/b          # 其中-p代表如果路径中有文件夹不存在，就会自动创建，也就是能同时创建路径中多个嵌套文件夹，比如a,b都可以不存在

# --------------------打印当前文件夹路径--------------------
pwd

# --------------------显示当前文件夹下所有子文件夹和子文件--------------------
ls

# --------------------创建文件夹快捷方式--------------------
ln -s abc/bc/c    # -s 代表symlink

# --------------------复制文件--------------------
cp src_file dst_file

# --------------------移动文件--------------------
mv src_file dst_file

# --------------------下载文件--------------------
curl -LO http://images.cocodataset.org/zips/train2017.zip

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

# --------------------判断某个目录下是否有指定文件夹--------------------
# if ; then  fi
# 注意if尾部需要有分号，fi代表end if
# [-d name]代表如果name存在且为目录
# [-a name]代表如果
if [ -d "build" ]; then
    rm -r build
fi
    

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

# --------------------打印当前文件目录树--------------------
sudo apt install tree
tree ssd       # 同时显示文件夹和文件
tree -L 1 ssd  # 只展示L1层级的文件夹和文件


