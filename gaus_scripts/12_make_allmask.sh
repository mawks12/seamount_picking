#!/bin/bash
#
#  compute VGG masks with heights ranging from 700 m to 2600, depths ranging from -500 to -8000
#
rm -f *.grd *.ps input

#
# 1 - get the seamount location and cut the grids
#
#
lonc="-130"
latc="0"
#
# set the region and rms region
#
lon0=$(echo "($lonc-.4)" | bc -l)
lonf=$(echo "($lonc+.4)" | bc -l)
lat0=$(echo "($latc-.4)" | bc -l)
latf=$(echo "($latc+.4)" | bc -l)
region="-R$lon0/$lonf/$lat0/$latf"
lon0=$(echo "($lonc-.15)" | bc -l)
lonf=$(echo "($lonc+.15)" | bc -l)
lat0=$(echo "($latc-.15)" | bc -l)
latf=$(echo "($latc+.15)" | bc -l)
rms_region="-R$lon0/$lonf/$lat0/$latf"

gmt grdcut SRTM15_V2.5.nc $region -Gtopo_d.grd -fg -Vq


for height in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600
do
  echo $height
  for topo0 in -1000 -1250 -1500 -1750 -2000 -2250 -2500 -2750 -3000 -3250 -3500 -3750 -4000 -4250 -4500 -4750 -5000 -5250 -5500 -5750 -6000 -7000 -8000
  do
  if [ $(($topo0+$height)) -lt -100 ]
  then
  echo $topo0
  ./make_seamountvgg.sh $height $topo0
  fi
  done
done
