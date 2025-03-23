#!/bin/bash
for dir in 0{1,4}*; do
    for i in {0..14}; do
        if [ -f $dir/$i/run.log ]; then
            python time-difference.py $dir/$i/run.log
        fi
    done
done > times.txt

cat times.txt
