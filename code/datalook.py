#!/usr/bin/env python
# coding: utf-8

# In[9]:


from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN
import simplekml
from smount_predictors import SeamountHelp
import xarray as xr


# In[17]:


training_mounts = pd.read_csv('out/handpick_test_coords.csv')
points = (-9.6, -6.9, -109.1, -106.6)
testing_data = SeamountHelp.readAndFilterGRD(Path('interfaces_exclude/swot_masked.grd'), points[:2], points[2:]).to_dataframe().reset_index()
training_mounts = pd.concat([training_mounts, catalog], ignore_index=True)
labeled_data = SeamountHelp.seamount_radial_match(testing_data, training_mounts)
catalog = pd.read_csv('data/all.xyhrdnc', sep=' ', names=['lon', 'lat', 'height','radius', 'depth', 'name', 'cat'])
catalog = catalog[['lat', 'lon', 'name', 'radius']]
catalog = catalog[(catalog['lat'] > points[0]) & (catalog['lat'] < points[1]) & (catalog['lon'] > points[2]) & (catalog['lon'] < points[3])]
px.scatter(labeled_data, x='lon', y='lat', color='Labels').show()


# In[20]:


training_set = xr.Dataset.from_dataframe(labeled_data)
training_set.to_netcdf('out/testing_set.nc')


# In[7]:


aall = pd.read_csv('data/all.xyhrdnc', sep=' ', names=[
    'lon', 'lat', 'height', 'radius', 'depth', 'name', 'catalouged'
])
import simplekml
kml = simplekml.Kml()
for i, row in aall.iterrows():
    pnt = kml.newpoint(name=row['name'], coords=[(row['lon'], row['lat'])])
    pnt.description = f'Height: {row["height"]}\nRadius: {row["radius"]}\nDepth: {row["depth"]}\nCatalouged: {row["catalouged"]}'
kml.save('data/all_cat.kml')


# In[9]:


get_ipython().system(' open data/all_cat.kml')


# ## TODO: fix testing data - catalouge does not contain many small seamounts

# In[ ]:




