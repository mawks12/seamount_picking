#!/bin/bash
#
#  mask out the uncharted areas with NaN
#
#gmt grdmath SID_V2.5.nc 1 MIN 0 NAN = tmp_mask.grd -fg
#gmt grdfilter tmp_mask.grd -R-180/180/-80/80 -D3 -Fb4 -V -fg -Ghit_mask.grd
#
#  add one more column to signify 1-charted, 0-uncharted
#
#pre="all"
#pre="good"
#pre="shallow"
#pre="short"
pre="tall"
echo $pre
rm charted.xyhrdnc
rm uncharted.xyhrdnc
#
gmt grdtrack ${pre}.xyhrdn -Ghit_mask.grd > /tmp/pre.tmp
awk '$6 == "1" { print $1,$2,$3,$4,$5,$7,"1"}' < /tmp/pre.tmp > charted.xyhrdnc
awk '$6 == "NaN" { print $1,$2,$3,$4,$5,$7,"0"}' < /tmp/pre.tmp > uncharted.xyhrdnc
cat charted.xyhrdnc uncharted.xyhrdnc > ${pre}.xyhrdnc
rm charted.xyhrdnc uncharted.xyhrdnc

