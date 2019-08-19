#!/bin/sh
# 本脚本用于同步本文件夹下所有git repo
# 同步之前需要评估：是否有大文件产生，如果有，需要先去除大文件才能同步到github
# 执行方式：从命令行进入该脚本目录，然后运行 sh sync.sh

# -------------------------------------------------------
del_ipch()
{
    for element in `find $1 -type d -name "ipch"`
    do          
        rm -rf $element          
    done
}

echo "starting pull all the assigned repo from github...."

cd ./simple_ssd_pytorch
git pull
cd ..
echo "finish pull 1/9..."

cd ./PythonCodingSkill
git pull
cd ..
echo "finish pull 2/9..."

cd ./cv_study
git pull
cd ..
echo "finish pull 3/9..."

cd ./Car_LaneLines_Detector
git pull
cd ..
echo "finish pull 4/9..."

cd ./lessonslearn
git pull
cd ..
echo "finish pull 5/9..."

cd ./CppStudy
git pull
cd ..
echo "finish pull 6/9..."

cd ./machine_learning_algorithm
git pull
cd ..
echo "finish pull 7/9..."

cd ./machine_learning_for_stock
git pull
cd ..
echo "finish pull 8/9..."

# 由于该仓库是从slcv继承过来的，直接pull下来会有巨大的.git文件夹，所以git clone采用--depth=1，
# 但接下来更新的pull就是问题，所以暂时不pull,只push，有空再pull
#cd ./cvpk
#git pull
#cd ..
#echo "finish pull 10/9..."


# -------------------------------------------------------
echo "starting push all the local update to github..."
cd ./simple_ssd_pytorch
path=$(pwd)
git add .
git commit -m 'update'
git push
cd ..
echo "------------finish push 1/9 in ${path}------------"

cd ./PythonCodingSkill
path=$(pwd)
git add .
git commit -m 'update'
git push
cd ..
echo "------------finish push 2/9 in ${path}------------"

cd ./cv_study
path=$(pwd)
git add .
git commit -m 'update'
git push
cd ..
echo "------------finish push 3/9 in ${path}------------"

cd ./Car_LaneLines_Detector
path=$(pwd)
git add .
git commit -m 'update'
git push
cd ..
echo "------------finish push 4/9 in ${path}------------"

cd ./lessonslearn
path=$(pwd)
git add .
git commit -m 'update'
git push
cd ..
echo "------------finish push 5/9 in ${path}------------"

cd ./CppStudy
path=$(pwd)          # 获得完整路径
echo "start to delete all the ipch document in ${path}"  # 删除ipch里边大文件
del_ipch $path       # 删除该路径下所有ipch文件夹 
git add .
git commit -m 'update'
git push
cd ..
echo "------------finish push 6/9 in ${path}------------"

cd ./machine_learning_algorithm
path=$(pwd)
git add .
git commit -m 'update'
git push
cd ..
echo "------------finish push 7/9 in ${path}------------"

cd ./machine_learning_for_stock
path=$(pwd)
git add .
git commit -m 'update'
git push
cd ..
echo "------------finish push 8/9 in ${path}------------"

cd ./cvpk
path=$(pwd)
git add .
git commit -m 'update'
git push
cd ..
echo "------------finish push 9/9 in ${path}------------"

# -------------------------------------------------------
echo 'synchronize finished!'
