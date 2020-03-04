#!/bin/bash

for topology in 'fracba' 'gnm' 'xgft' 'dcell' 'spdcell'; do
# for topology in 'zoo'; do
    for k in 1 2 3 4 8; do
        for s in 400; do
            for a in 40; do
                echo "`date`: setcover experiments $topology k $k s $s a $a"
                echo "`date`: setcover experiments $topology k $k s $s a $a" >> setcover.log
                python explore-setcover-par-lock.py --$topology --k $k --maxs $s --server --T1000  --seta $a --seedfile --nprocs 54
                echo "`date`: Done with setcover $topology k $k s $s a $a"
                echo "`date`: Done with setcover $topology k $k s $s a $a" >> setcover.log
            done
        done
    done    
done