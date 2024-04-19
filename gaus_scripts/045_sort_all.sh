#!/bin/bash
# 
rm tall.xyhrdn
rm short.xyhrdn
rm shallow.xyhrdn
rm good.xyhrdn
#
   awk '{print $6 }' < all.xyhrdn > test.txt
#
for smt in `cat test.txt`
do
#
#  extract the line corresponding to this seamount name
#  and get its height
#
  grep $smt all.xyhrdn > /tmp/save.txt
  height=$( awk '{print $3}' < /tmp/save.txt )
  d_min=$( awk '{printf ("%d\n",$3+$5) }' < /tmp/save.txt )
#  echo $height $d_min
  if [ $height -gt 2500 ]
     then
     cat /tmp/save.txt >> tall.xyhrdn
  elif [ $height -lt 700 ]
     then
     cat /tmp/save.txt >> short.xyhrdn
  else
     if [ $d_min -le -800 ]
        then
            cat /tmp/save.txt >> good.xyhrdn
        else
            cat /tmp/save.txt >> shallow.xyhrdn
     fi
  fi
done
