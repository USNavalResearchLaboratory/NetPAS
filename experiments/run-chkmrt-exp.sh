#!/bin/bash

for topology in 'gnm' 'xgft' 'dcell' 'spdcell'; do
# for topology in 'zoo'; do
    for k in 3; do
        for s in 400; do
            echo "`date`: CHKMRT-sum experiments $topology k $k s $s with seedfile"
            echo "`date`: CHKMRT-sum experiments $topology k $k s $s with seedfile" >> chkmrt.log
            python exp-chkmrt.py --sum --$topology --k $k --maxs $s  --T1000 --server --seedfile
            echo "`date`: Done with CHKMRT-sum experiments $topology k $k s $s with seedfile"
            echo "`date`: Done with CHKMRT-sum experiments $topology k $k s $s with seedfile" >> chkmrt.log

            echo "`date`: CHKMRT-max experiments $topology k $k s $s with seedfile"
            echo "`date`: CHKMRT-max experiments $topology k $k s $s with seedfile" >> chkmrt.log
            python exp-chkmrt.py --maxcvxpy --$topology --k $k --maxs $s  --T1000 --server --seedfile
            echo "`date`: Done with CHKMRT-max experiments $topology k $k s $s with seedfile"
            echo "`date`: Done with CHKMRT-max experiments $topology k $k s $s with seedfile" >> chkmrt.log

        done
    done
done