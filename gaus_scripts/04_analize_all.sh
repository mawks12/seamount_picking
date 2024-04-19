#!/bin/bash
# 
#  compiles all the relevant data for each seamount
#  lon, lat, height, radius, base_depth, name
#
rm all.xyhrdn
#cd misfit_all
#ls > ../test.txt
#cd ..
#
# edit the file test.txt and remove the suffix
#
#
#  vi and remove .txt from names
#
for smt in `cat test.txt`
do
#
# 1 - find the height with the smallest vgg misfit
#
  sort -n misfit_all/$smt.txt | head -1 > /tmp/best.txt
  height=$(cat /tmp/best.txt | awk '{print $2}')
#
# 2 - get the seamount location and cut the grids
#
  grep $smt centered_all.xy | head -1 > center.xy
  lonc=$( awk '{print $2}' < center.xy )
  latc=$( awk '{print $3}' < center.xy )
#
# set the region and cut the data
#
  flon=$(echo "(1./c($latc/57.29))" | bc -l)
  lon0=$(echo "($lonc-.4*$flon)" | bc -l)
  lonf=$(echo "($lonc+.4*$flon)" | bc -l)
  lat0=$(echo "($latc-.4)" | bc -l)
  latf=$(echo "($latc+.4)" | bc -l)
  region="-R$lon0/$lonf/$lat0/$latf"
  gmt grdcut SRTM15_V2.5.nc $region -Gtopo_d.grd -fg -Vq
#
#  3 - estimate the shape of each seamount and the mean ocean depth
#
  topo0=$(gmt grdinfo -L1 topo_d.grd | grep median | awk '{print $3}')
#
#  the magic number 7.278 is 3 sigma
#
  radius=$(echo "($height/1000)*7.278" | bc -l)
#
  echo $lonc $latc $height $radius $topo0 $smt >> all.xyhrdn
#
done
