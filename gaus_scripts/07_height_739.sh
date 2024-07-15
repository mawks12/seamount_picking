#!/bin/bash
# 
rm -f *.grd *.ps input smt_models.xyz
rm height_739.txt
#
for smt in `cat KW_lists/739_smts.txt`
do
#
#  find the height with the smallest vgg misfit
#
  sort -n misfit_orig/$smt.txt | head -1 > /tmp/best.txt
  height=$(cat /tmp/best.txt | awk '{print $2}')
  lon=$(cat /tmp/best.txt | awk '{print $3}')
  lat=$(cat /tmp/best.txt | awk '{print $4}')
  echo $smt $height >> height_739.txt
done
