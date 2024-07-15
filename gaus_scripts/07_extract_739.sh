#!/bin/bash
# 
rm smt_739.txt
#
for smt in `cat KW_SIO_lists/739_smts.txt`
do
#
#  find the height with the smallest vgg misfit
#
  sort -n misfit_all/$smt.txt | head -1 > /tmp/best.txt
  rms=$(cat /tmp/best.txt | awk '{print $1}')
  height=$(cat /tmp/best.txt | awk '{print $2}')
  lonc=$(cat /tmp/best.txt | awk '{print $3}')
  latc=$(cat /tmp/best.txt | awk '{print $4}')
  flon=$(echo "(1./c($latc/57.29))" | bc -l)
  lon0=$(echo "($lonc-.05*$flon)" | bc -l)
  lonf=$(echo "($lonc+.05*$flon)" | bc -l)
  lat0=$(echo "($latc-.05)" | bc -l)
  latf=$(echo "($latc+.05)" | bc -l)
  region="-R$lon0/$lonf/$lat0/$latf"
  gmt grdcut topo_25.1.nc $region -Gtopo_d.grd -fg -Vq
  summit_meas=$(gmt grdinfo topo_d.grd | grep v_max | awk '{print $5}')
#  echo $summit_meas

  rms=$(cat /tmp/best.txt | awk '{print $1}')
#
  grep $smt all.xyhrdn > /tmp/one.xyhrdn
  base=$(cat /tmp/one.xyhrdn | awk '{print $5}')
  summit_vgg=$(cat /tmp/one.xyhrdn | awk '{print $5+$3}')
#
# get the predicted and measured summit depth
#
  awk '{ print $3, $4 }' < /tmp/best.txt > /tmp/best.ll
  summit_pred=$(gmt grdtrack /tmp/best.ll -Gtopo_predict.nc | awk '{print $3}')
#  summit_meas=$(gmt grdtrack /tmp/best.ll -Gtopo_25.1.nc | awk '{print $3}')
  echo $summit_meas $summit_pred $summit_vgg $base $height $rms >> smt_739.txt
done
