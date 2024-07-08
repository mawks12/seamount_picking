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
import os

print(os.getcwd())
training_bounds = SeamountHelp.readKMLbounds(Path('data') / 'Seamount_training_zone.kml')
points = SeamountHelp.read_seamount_centers(Path('data') / 'Seamount_training_zone.kml')
srtm = SeamountHelp.readAndFilterGRD(Path('data') / 'srtm15_V2.5.nc', training_bounds[:2], training_bounds[2:])

points.drop(index='mh37', inplace=True)

parameter = -3000
filtered_srtm = pd.read_csv('data/clustered_seamounts.csv')
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import StandardScaler

# Assuming filtered_srtm.to_dataframe().reset_index()[['lat', 'lon', 'z']] is your dataset
X = filtered_srtm[['lat', 'lon']]
X = X.dropna()

# Standardize the data
X_scaled = X.copy()
# Train a KMeans clusterer
kmeans = KMeans(n_clusters=100, algorithm='elkan')
kmeans.fit(X_scaled)


cluster_labels = kmeans.fit_predict(X)
X['cluster'] = cluster_labels
X.to_csv('data/clustered_seamounts.csv')
