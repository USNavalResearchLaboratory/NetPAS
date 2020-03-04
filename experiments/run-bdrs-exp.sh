#!/bin/bash

for topology in 'fracba' 'gnm' 'xgft' 'dcell' 'spdcell'; do
# for topology in 'zoo'; do
    for k in 1 2 4 8; do
        for s in 400; do
            for ov in '' '--noov'; do

                echo "`date`: BDRS experiments $topology s $s $ov k $k eqimp in parallel"
                echo "`date`: BDRS experiments $topology s $s $ov k $k eqimp in parallel" >> bdrs.log
                python exp-bdrs.py --$topology $ov --k $k --maxs $s --eqimp --T1000 --seedfile --server
                echo "`date`: Done with BDRS $topology s $s $ov k $k eqimp in parallel"
                echo "`date`: Done with BDRS $topology s $s $ov k $k eqimp in parallel" >> bdrs.log


                echo "`date`: BDRS experiments $topology s $s $ov k $k pathimp in parallel"
                echo "`date`: BDRS experiments $topology s $s $ov k $k pathimp in parallel" >> bdrs.log
                python exp-bdrs.py --$topology $ov --k $k --maxs $s --pathimp --T1000 --seedfile --server
                echo "`date`: BDRS experiments BDRS $topology s $s $ov k $k pathimp in parallel"
                echo "`date`: BDRS experiments BDRS $topology s $s $ov k $k pathimp in parallel" >> bdrs.log

            done
        done
    done    
done