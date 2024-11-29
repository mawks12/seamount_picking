#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import pandas as pd
from smount_predictors import SeamountTransformer, SeamountHelp, SeamountCVSplitter
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import plotly.express as px
import numpy as np


# In[2]:


pipe = Pipeline([
    ('trans', SeamountTransformer()),
    ('predictor', SVC(kernel='rbf', class_weight='balanced'))  # previous grid searches have found optimal C of 1
])

param_grid = {
    'predictor__C': np.linspace(1, 10),
    'trans__sigma': np.linspace(0.1, 2)
    }

search = GridSearchCV(estimator=pipe, param_grid=param_grid, n_jobs=-1, cv=SeamountCVSplitter(5), verbose=3)


# In[3]:


points = SeamountHelp.readKMLbounds(Path('data/seamount_training_zone.kml'))
data = SeamountHelp.readAndFilterGRD(Path('data') / 'training_data_new.nc')
X = data.to_dataframe().reset_index()[['lat', 'lon' , 'z']]


# In[ ]:


y = data.to_dataframe().reset_index()['Labels']
search.fit(X, y)
print(f'train score: {search.score(X, y)}')


# In[4]:


import pickle
from sklearn.cluster import DBSCAN
from smount_predictors.src.SeamountHelp import PipelinePredictor

full_pipeline = PipelinePredictor(search, DBSCAN(eps=0.00029088820866630336, min_samples=4, metric='haversine'))
pickle.dump(full_pipeline, open('out/3d_model.pkl', 'wb'))


# In[ ]:


import xarray as xr
import pygmt


X['Labels'] = pipe.predict(X)
predictions = xr.Dataset.from_dataframe(X.set_index(['lat', 'lon'])).set_coords(['lon', 'lat'])


# In[ ]:


points = (-19.20600998877477, -15.16349705205003, -117.7208544442338, -110.2604021311965)
srtm = xr.open_dataset('data/SRTM15_V2.5.nc')
fig = pygmt.Figure()
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT_LABEL="15p,Helvetica,black", FONT_ANNOT="15p")
pygmt.config(FONT_ANNOT_PRIMARY="15p,Helvetica,black")

############################ SWIR ##################################
pygmt.makecpt(cmap="haxby", series=[-4000, -1500, 1], background='o')
fig.grdimage(
    grid = xr.DataArray(srtm.z, coords=(srtm.lat, srtm.lon)),
    # shading = shade_SWIR,
    projection="M90c",
    region = [points[2], points[3], points[0], points[1]],
    frame=["WSrt", "xa0.5", "ya0.5"],
    cmap = True,
    # shading = pygmt.grdgradient(xr.DataArray(srtm.z, coords=(srtm.lat, srtm.lon)), direction='a', normalize=True)
    )
fig.grdcontour(
    grid=predictions['Labels'],
    annotation="200+f12p", 
    interval=1,
    pen = "1p, black",
    limit=[-3400, 2800]
    )
fig.show()

