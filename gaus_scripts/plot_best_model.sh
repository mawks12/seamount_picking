#!/bin/bash
  
rm -f *.grd *.ps input

rm misfit/KW-13664.txt
predict_seamount.sh KW-13664 1950
region=$(gmt grdinfo topo_diff.grd -I-)
map_topo.csh 0 $region topo_d 
map_topo.csh 0 $region topo 
map_topo.csh 0 $region topo_diff 
map_grav.csh 60 $region VGG 
map_grav.csh 60 $region VGG_d
map_grav.csh 60 $region VGG_m
map_grav.csh 60 $region VGG_diff
map_grav.csh 60 $region VGG_diff_m
