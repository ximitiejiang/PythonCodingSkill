#!/bin/sh
# -------------------------------------------------------
echo 'starting pull all the assigned repo from github....'

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


