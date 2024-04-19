#!/bin/bash
# 
rm -f *.grd *.ps input smt_models.xyz
#
for smt in `cat test.txt`
do
#  echo $smt.txt
#
#  find the height with the smallest vgg misfit
#
  sort -n misfit_all/$smt.txt | head -1 > /tmp/best.txt
  height=$(cat /tmp/best.txt | awk '{print $2}')
  if [ $height -lt 2600 -a $height -gt 700 ]; then
    echo $smt $height
    best_seamount.sh $smt $height
    d_min=$(gmt grdinfo topo_mask.grd | grep v_max | awk '{ printf ("%d\n", $5) }')
#
#  if the seamount is too shallow then reset the height 
#
    if [ $d_min -gt -100 ]; then
      ht2=$(echo "($height - ($d_min+100))" | bc -l)
#      echo $height $ht2
      best_seamount.sh $smt $ht2
      d_min2=$(gmt grdinfo topo_mask.grd | grep v_max | awk '{ printf ("%d\n", $5) }')
#      echo $smt $d_min $d_min2 $height $ht2
#      region=$(gmt grdinfo topo_mask.grd -I-)
#      map_topo.csh 0 $region topo_mask
#      open topo_mask.pdf
#      read
    fi
    gmt grd2xyz topo_mask.grd -s >> smt_models.xyz
   fi
done
