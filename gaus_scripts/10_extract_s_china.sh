#!/bin/bash
 # 
rm shallow.xydn
rm deeper.xydn
#
# start with all the seamounts
#
awk '{ print $2, $3, $4 }' < shallow.xyn > tmp.xyn
cat tall.xyn tmp.xyn good.xyn > all.xyn
#
# first limit the area using awk
#
   awk ' {  if($1 < 120 && $1 > 109 && $2 > 9 && $2 < 22) print $3 }' < all.xyn > test2.txt
#
for smt in `cat test2.txt`
do
#
#  find the height with the smallest vgg misfit
#
  sort -n misfit_orig/$smt.txt | head -1 > /tmp/best.txt
  height=$(cat /tmp/best.txt | awk '{print $2}')
  lon=$(cat /tmp/best.txt | awk '{print $3}')
  lat=$(cat /tmp/best.txt | awk '{print $4}')
  best_seamount.sh $smt $height
  d_min=$(gmt grdinfo topo_mask.grd | grep v_max | awk '{ printf ("%d\n", $5) }')
  echo $lon $lat $d_min $smt
  if [ $d_min -le -400 ]
     then
         echo $lon $lat $d_min $smt >> deeper.xydn
     else
         echo $lon $lat $d_min $smt >> shallow.xydn
  fi
done
