#!/bin/bash

del_ipch(){
    for element in `find $1 -type d -name "ipch"`
    do          
        rm -rf $element          
    done
}

root_dir='./'
del_ipch $root_dir

# ll |grep "^-"|wc -l    # ll是ls -l的别名,表示显示文件，按照长格式l, grep "^-"是过滤，只保留一般文件，^d则是只保留目录， wc -l是统计输出信息行数，这里就是经过滤后的一般文件的行数
