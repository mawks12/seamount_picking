#!/bin/bash
#
#  script to use the seamount location, regional depth and VGG to make a seamount model
#
# gravity calculation constants
rhow="1030" # kg/m3
rhoc="2800" # kg/m3
rhom="3300" # kg/m3
G="0.0000000000667" # gravitational constant
height=$2
#
# 1 - get the seamount location and cut the grids
#
grep $1 centered_all.xy | head -1 > center.xy
lonc=$( awk '{print $2}' < center.xy )
latc=$( awk '{print $3}' < center.xy )
#
# set the region and rms region
#
flon=$(echo "(1./c($latc/57.29))" | bc -l)
lon0=$(echo "($lonc-.4*$flon)" | bc -l)
lonf=$(echo "($lonc+.4*$flon)" | bc -l)
lat0=$(echo "($latc-.4)" | bc -l)
latf=$(echo "($latc+.4)" | bc -l)
region="-R$lon0/$lonf/$lat0/$latf"
lon0=$(echo "($lonc-.15*$flon)" | bc -l)
lonf=$(echo "($lonc+.15*$flon)" | bc -l)
lat0=$(echo "($latc-.15)" | bc -l)
latf=$(echo "($latc+.15)" | bc -l)
rms_region="-R$lon0/$lonf/$lat0/$latf"
#gmt grdcut SRTM15_V2.5.nc $region -Gtopo_d.grd -fg -Vq
gmt grdcut topo_25.1.nc $region -Gtopo_d.grd -fg -Vq
gmt grdcut curv_32.1.nc $region -Gtmp.grd -fg -Vq
gmt grdsample tmp.grd -Rtopo_d.grd -GVGG_d.grd -fg -Vq
#
#  2 - estimate the shape of each seamount and the mean ocean depth
#
topo0=$(gmt grdinfo -L1 topo_d.grd | grep median | awk '{print $3}')
#echo "base depth" $topo0
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
gmt grdfilter tmp.grd -Fg16 -D2 -GVGG_m.grd -Vq
VGG0=$(gmt grdinfo -L1 VGG_d.grd | grep median | awk '{print $3}')
VGGf=$(gmt grdinfo -L1 VGG_d.grd | grep v_max | awk '{print $5}')
VGG_height=$(echo "($VGGf)-($VGG0)" | bc -l)
#echo "VGG height" $VGG_height
#
# 3 - make the difference grids
#
gmt grdmath $rms_region topo_d.grd topo.grd SUB = topo_diff.grd -Vq
gmt grdmath $rms_region VGG_d.grd VGG_m.grd SUB = VGG_diff_m.grd -Vq
#gmt grdinfo -L1 topo_diff.grd | grep median
L1_t=$(gmt grdinfo -L1 topo_diff.grd | grep median | awk '{print $5}')
L1_s=$(gmt grdinfo -L1 VGG_diff_m.grd | grep median | awk '{print $5}')
#gmt grdinfo -L1 VGG_diff_m.grd | grep median
echo $L1_s $height $lonc $latc $L1_t >> misfit/$1.txt
echo $height $L1_t $L1_s
#
#  mask the model at a radius of 2 sigma to create the fake data
#
#echo $lonc $latc > c.xy
#echo $radius
#gmt grdmath c.xy LDIST $radius_mask LE 0 NAN = mask.grd -fg -Rtopo.grd
#gmt grdmath mask.grd topo.grd MUL = topo_mask.grd
