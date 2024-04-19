#!/bin/bash
#
#  compute models with heights ranging from 700 m to 2600
#
rm -f *.grd *.ps input
#
for smt in `cat test.txt` # list of seamounts, still uncertain where it is crossreferencing
do
  echo $smt.txt
  rm misfit/$smt.txt
  for height in 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600
  do
  predict_seamount.sh $smt $height # this produces the misfit files referenced in other scripts
  done
done
