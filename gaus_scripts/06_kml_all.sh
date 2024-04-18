#!/bin/bash
 # 
rm -f *.grd *.ps input smt_models.xyz
rm tall.kml
rm short.kml
rm shallow.kml
rm good.kml
rm charted.kml
rm uncharted.kml
#
awk '{print $1,$2,$6 }' < shallow.xyhrdnc | gmt gmt2kml -Fs -Gred+f -Nt > shallow.kml
awk '$7 == "1" {print $1,$2,$6 }' < shallow.xyhrdnc | gmt gmt2kml -Fs -Ggreen+f -Nt > shallow_c.kml
awk '$7 == "0" {print $1,$2,$6 }' < shallow.xyhrdnc | gmt gmt2kml -Fs -Gred+f -Nt > shallow_u.kml
awk '{print $1,$2,$6 }' < tall.xyhrdnc | gmt gmt2kml -Fs -Gyellow+f -Nt > tall.kml
awk '{print $1,$2,$6 }' < short.xyhrdnc | gmt gmt2kml -Fs -Gblue+f  -Nt > short.kml
awk '{print $1,$2,$6 }' < good.xyhrdnc | gmt gmt2kml -Fs -Gtan+f -Nt > good.kml
awk '$7 == "1" {print $1,$2,$6 }' < all.xyhrdnc | gmt gmt2kml -Fs -Ggreen+f -Nt > charted.kml
awk '$7 == "0" {print $1,$2,$6 }' < all.xyhrdnc | gmt gmt2kml -Fs -Gred+f -Nt > uncharted.kml

