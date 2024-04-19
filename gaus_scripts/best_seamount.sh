#!/bin/bash
#
#  script to make a masked model of the best seamount
#
height=$2
#
# 1 - get the seamount location and cut the grids
#
grep $1 centered_all.xy | head -1 > center.xy
lonc=$( awk '{print $2}' < center.xy )
latc=$( awk '{print $3}' < center.xy )
#
# set the region and cut the data
#
flon=$(echo "scale=5; ( 1./c($latc/57.29))" | bc -l)
#echo $flon
lon0=$(echo "($lonc-.4*$flon)" | bc -l)
lonf=$(echo "($lonc+.4*$flon)" | bc -l)
lat0=$(echo "($latc-.4)" | bc -l)
latf=$(echo "($latc+.4)" | bc -l)
region="-R$lon0/$lonf/$lat0/$latf"
#echo $region
gmt grdcut SRTM15_V2.5.nc $region -Gtopo_d.grd -fg -Vq
#
#  2 - estimate the shape of each seamount and the mean ocean depth
#
topo0=$(gmt grdinfo -L1 topo_d.grd | grep median | awk '{print $3}')
#echo "base depth" $topo0
#
#  the magic number 7.278 is 3 sigma
#
radius=$(echo "($height/1000)*7.278" | bc -l)
#
#  make the mask radius 1.5 sigma
#
radius_mask=$(echo "($height/1000)*3.639" | bc -l)
echo $lonc $latc $radius $height > input
#
gmt grdseamount -fg input -Rtopo_d.grd -Gtopo.grd -Cg -Z$topo0 -Vq
#
#  mask the model at a radius of 2 sigma to create the fake data
#
echo $lonc $latc > c.xy
gmt grdmath c.xy LDIST $radius_mask LE 0 NAN = mask.grd -fg -Rtopo.grd
gmt grdmath mask.grd topo.grd MUL = topo_mask.grd
d_min=$(gmt grdinfo topo_mask.grd | grep v_max | awk '{ print $5 }')
#echo $1 $height $radius_mask $topo0 $d_min
