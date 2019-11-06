# Python check sheet


### 关于list
```
a = [1,2,3]          #创建
a.append(2)　　　　　＃扩展　　　　
a.extend(2)　　　　　＃扩展　　
a.insert(-1, [1,2])  # 任意位置插入
print(a[2])

a[2:8:2]            # 切片
s[2:-1]

a + b                # list的合并
a * 3                # list的重复合并，而不是元素计算
a = [i*3 for i in a] # 用列表推导式对list的元素进行计算

func(*lst, **dct)    # list的解包，可以用来给函数提供位置参数，dict的解包则可提供关键字参数
```

### 关于numpy
```
a = np.array([1,2])
a = np.zeros((3,5))  # 双括号
a = np.ones((3,5))
a = np.empty((3,5))

b = np.zeros_like(a)
b =

np.arange(1,10,2) 
```


### 关于dict
```
a = {'a':1, 'b':2}
a = dict(a=1, b=2)

a.get("a", None)
a.update(c=4)
a.set_default('d', 100)  # 如果存在则剔除，如果不存在则创建
a.pop('b')

d = {**d1, **d2}    # 采用解包，最快合并两个字典

```


### 关于tuple
1. tuple与list区别在于tuple的元素值不能改变
```
a = (2,4)
```


### 关于string
1. python不区分双引号和单引号，都可以表示单个字符或者字符串
```

```
2. 字符串的操作函数
```
s.upper()
s.lower()
s.title()
s.split()
s.strip()       # 去除两边的字符，一般不太用，直接切片完成即可
s.startswith()
s.endswith()
s.islower()
s.isdigit()
s.isalpha()
s.isspace()
```


### 关于set
1. set是集合，最大的好处一个是自动去除重复数据，另一个是可以方便使用集合运算符
```
s1 = set(lst1)       # 创建set
s2 = set([3,3,7,4])

s1 + s2    # 相加
s1 - s2    # 相减
s1 | s2    # 相或
s1 & s2　　# 相与　
s1 ^ s2    # 异或
```


### 关于打印输出
1. python不区分双引号和单引号
```
print('a = %.3f, b=%s, c=%d'%(a, b, c))          # 第一种方式(类似c++)，f代表浮点数，s代表字符串，d代表整数
print('a = {.3f}, b={}, c={}'.format(a, b, c))   # 第二种方式，python的format关键字
```


### 关于数据类型和类型转换
0. 采用a.dtype可以查看类型
```
a.dtype    # 查看数据类型，只能对单个数值判断是int/float/..
type(a)    # 查看类型，包括是list/str/tuple/class
```
1. 数据类型在python原生里边有：int, float, 转换方式就是
```
int(a)
float(b)
```
2. 数据类型在numpy中有：np.uint8, np.int8, np.float32, np.float64, 转换方式
```
a = np.array(b).astype(np.float32)
```


### 关于基本语句
1. 基本逻辑控制语句
```
for i in range(10)：
    print(i)
    
while a >0:
    print(a)
    a -= 1


```            
2. 重要的with语句： 能够帮我们自动释放变量内存
```
with open(paht, 'rb') as f:     # 用with打开文件，退出with之后就自动关闭文件
    lines = f.readlines()

with get_engine() as engine, get_logger() as logger:  # 用with声明变量，退出with之后变量就自动销毁
    data = engine(logger)
```
3. 重要的try..except语句：可以用来在异常的时候防止程序退出而继续执行
```
try：
    get_engine(data)
except:
    print(data)
```

### 关于基本运算符
1. 逻辑运算符： 与，或，非(and, or, !)
```
a and b
a or b
!a
```
2. 按位运算符: &, |, ^, ~, <<, >>
```
a & b
a << 20    #左移20位，相当于乘以2^20
```
3. 集合运算符：先转化为set，然后进行集合运算，符号跟按位运算符一样
```
s1 & s2
s1 | s2
s1 ^ s2
```


### 关于矩阵乘法
1. 对于array来说：跟mat唯一差别是*运算符号的差别，一个是按位，一个是矩阵，如果怕混乱，可用两个函数这样不会错mutiply按位，dot矩阵
```
ar1 * ar2             # 按位乘
np.multiply(ar1, ar2) # 按位乘
np.dot(ar1, ar2)      # 矩阵乘法(行乘以列求和)
```

2. 对于mat来说
```
mt1 * mt2             # 矩阵乘法(变了)
np.multiply(mt1, mt2) # 按位乘
np.dot(mt1, mt2)      # 矩阵乘法
```


### 关于随机数
1. 生成随机数
random.randn()


2. 提取一个随机数
random.choice(lst)

3. 随机打乱一组数
random.permute(lst)


### 关于迭代器
1. 迭代对象：有2中迭代对象，一种是可以切片的，一种是

2. 迭代器的方便用法
```
for i in data:
    print(i)
    
for data, value in zip(d.keys(), d.values()):
    print(data, value)
```


### 关于复制(浅拷贝和深拷贝)




### 关于排序 (重要！！！)
1. list的排序，用python原生的sorted(a)不影响源数据，或者a.sort()影响源数据
```
sorted(a)  #复制排序　　
a.sort()   #直接排序
a[::-1]    #逆序
```

2. numpy的排序，


### 关于筛选 (重要！！！)





### 关于维度axis和广播机制 (重要！！！)



### 关于None
1. 抓住一点：None/0/[]/()/{}/False都是等效的
```
if a:    # 如果a不为None，不为空，都可以
    print(a)

if a is not None:  # 这种判断比a!=None更好，因为如果a是[]空数组，则会报错
    return
```

### 关于partial()和map()
1. 这两个函数往往一起使用，非常便于写出分简洁的代码，省去for循环和其中大量的切片操作
```
def func(aa, bb, cc, d, e, f):
    return

pfunc = partial(func, d, e, f)               # 先用partial绑定固定的参数 
result = pfunc(aa_list, bb_list, cc_list)    # 再用map提取循环的参数
```


### 关于数据的堆叠
1. 如果是单组数据的重复复制后的堆叠，采用


2. 如果是多组不同数据的堆叠，采用np.concatenate(), np.stack()
```
```


### 关于装饰器



### 关于绘图命令
1. 采用matplotlib进行绘图


2. 采用cv2进行图像处理



### 关于文件路径以及读取和写入文件
1. 路径操作
```
os.path.basename(path)  # 文件名
os.path.dirname(path)   # 当前文件夹名(相对于__main__)
os.path.abspath(path)   # 绝对路径
os.path.isdir(path)
os.path.isfile(path)
os.pardir              # 代表父文件夹，也就是'..'
os.curdir              # 代表当前文件夹，也就是'.'
```
2. 相对路径
- (相对main文件路径)相对路径都是相对于__main__文件，'./'是相对于main文件的当前目录，'../'是相对于main文件的上级目录
- (相对所在文件路径)如果需要获得相对所在文件的路径，则可以利用__file__变量，path = os.path.dirname(os.path.abspath(__file__))

3. 读写文件
```
with open(path, 'rb') as f:
    lines = f.readlines()
    for line in lines:
        line.split()


with open(path, 'wb') as f:
    data = 'hello'
    f.write(data)
```

### 关于    