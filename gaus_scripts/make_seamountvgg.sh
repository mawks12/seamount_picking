#!/bin/bash
#
#  script to use the seamount location, regional depth and VGG to make a seamount model
#
# gravity calculation constants
rhow="1030" # kg/m3
rhoc="2800" # kg/m3
rhom="3300" # kg/m3
G="0.0000000000667" # gravitational constant
height=$1
topo0=$2 #mean ocean depth
#
lonc="-130"
latc="0"
#
#  the magic number 7.278 is 3 sigma
#
radius=$(echo "($height/1000)*7.278" | bc -l)
radius_mask=$(echo "($height/1000)*4.852" | bc -l)
echo $lonc $latc $radius $height > input
#
gmt grdseamount -fg input -Rtopo_d.grd -Gtopo.grd -Cg -Z$topo0 -Vq
#
#  now make the model VGG
#
diff=$(echo "$rhoc-$rhow" | bc -l)
gmt gravfft topo.grd -fg -Fv -D$diff -Gtmp.grd -Vq
gmt grdfilter tmp.grd -Fg16 -D2 -GVGG_m1.grd -Vq
gmt grdfilter VGG_m1.grd -Fg106+h -D2 -GVGG_m.grd -Vq
mv VGG_m.grd ./vggmask/VGG_${height}_${topo0:1:4}.grd
