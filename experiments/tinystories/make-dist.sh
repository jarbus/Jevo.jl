#!/bin/bash

for dir in 0{1,2,3,7}*; do
    for i in $dir/*; do
        grep -oP '(?<=\|).*?(?=\|)' $i/run.log | tr '±' ',' > $i/dist.csv
    done
done
