#!/usr/bin/env python3

import os
import re
import sys
from pathlib import Path
import simplekml
import pandas as pd


filepath = sys.argv[1]
filepath = Path(filepath)
with open(filepath, 'r') as fin:
    txt = fin.readlines()
    for ind, line in enumerate(txt):
        line = re.sub(r'^\s+', '', line)
        line = re.sub(r'°$', '', line)
        line = re.sub(r'°\s{2,3}', ',', line)
        txt[ind] = line
    txt = "".join(txt)
    with open('newPicked.csv', 'w') as fout:
        fout.write(txt)

filtered = pd.read_csv('newPicked.csv')

kml = simplekml.Kml()
earth = kml.newfolder(name='MH_seamounts')
mountstyle = simplekml.Style()
mountic = simplekml.Icon(href="http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png")
mounticon = simplekml.IconStyle(color='ffffff20', scale=0.8, icon=mountic)
mounticon.color = 'ffffff20'
mountstyle.iconstyle = mounticon
for ind, row in filtered.iterrows():
    ind = ind + 2
    pnt = earth.newpoint(name=f'MH_{ind}', coords=[(row.lon, row.lat)])
    pnt.style = mountstyle
kml.save('mounts.kml')
os.system('open mounts.kml')
