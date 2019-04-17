#!/bin/sh

num=0
while(( $num -lt 5 ))
do
    echo "$num"
    let "num++"
done