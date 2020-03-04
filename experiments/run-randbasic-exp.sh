#!/bin/bash

for topology in 'fracba' 'gnm' 'xgft' 'dcell' 'spdcell'; do
# for topology in 'zoo'; do
    for k in 1 2 4 8; do
        for s in 400; do
            echo "`date`: randbasic experiments 1000 $topology k $k s $s"
            echo "`date`: randbasic experiments 1000 $topology k $k s $s" >> randbasic.log
            python exp-randbasic.py --$topology --k $k --maxs $s --T1000 --seedfile --server
            echo "`date`: Done with randbasic 1000 $topology k $k s $s"
            echo "`date`: Done with randbasic 1000 $topology k $k s $s" >> randbasic.log
        done
    done
done