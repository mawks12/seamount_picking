#!/bin/bash
 
rm best_KW.txt
 
for smt in `cat KW_SIO_lists/KW_good.txt`
do
  echo $smt.txt
#
#  find the height with the smallest vgg misfit
#
  sort -n misfit/$smt.txt | head -1 >> best_KW.txt
done
