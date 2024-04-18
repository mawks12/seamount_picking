#!/bin/bash
 
rm best_SIO.txt
 
for smt in `cat KW_SIO_lists/SIO_all.txt`
do
  echo $smt.txt
#
#  find the height with the smallest vgg misfit
#
  sort -n misfit/$smt.txt | head -1 >> best_SIO.txt
done
