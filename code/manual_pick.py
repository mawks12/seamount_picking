#!/usr/bin/env python
# coding: utf-8

# In[21]:


from pathlib import Path
import simplekml
import numpy as np


# In[22]:


coords = Path('data/handpick_test_coords.csv')
kml = simplekml.Kml()
with coords.open() as f:
    lls = f.read().split()
    lls = [float(ll[:-1]) for ll in lls]
lls = np.array(lls).reshape(-1, 2)
lls


# In[25]:


names = []
for ind, ll in enumerate(lls):
    kml.newpoint(coords=[(ll[1], ll[0])], name=f'mh_{ind + 1}')
    names.append(f'mh_{ind + 1}')
kml.save('out/handpick_test_coords.kml')


# In[24]:


import pandas as pd

df = pd.DataFrame({'name': names, 'lat': lls[:, 0], 'lon': lls[:, 1]})
df.to_csv('out/handpick_test_coords.csv', index=False)


# In[21]:


from smount_predictors import SeamountHelp
new_coords = pd.read_csv('out/handpick_test_coords.csv')
old_coords = pd.read_csv('data/all.xyhrdnc', sep=' ', names=['lon', 'lat', 'height', 'radius', 'dept', 'name', 'cat'])
old_coords = old_coords[['lat', 'lon', 'radius', 'name']]
points = SeamountHelp.readKMLbounds(Path('data/seamount_training_zone.kml'))
old_coords = old_coords[(old_coords['lat'] > points[0]) & (old_coords['lat'] < points[1]) & (old_coords['lon'] > points[2]) & (old_coords['lon'] < points[3])]
train_coords = pd.concat([old_coords, new_coords], ignore_index=True)
train_coords.to_csv('out/train_mounts.csv')


# In[ ]:




