#!/bin/bash

for dir in 0{1,2,3}*; do
    for i in $dir/*; do
        if [ -f $i/run.log ]; then
            trial=$(basename $i)
            grep -B 1 "gen=40" $i/run.log | python extract-last-gen-information.py $dir $i
        fi
    done
done > last-gen-information.json
