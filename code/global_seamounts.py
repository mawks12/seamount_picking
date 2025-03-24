"""Script for applying seamount model to global vgg mapping."""

# /usr/bin/env python
# coding: utf-8
import os
from pathlib import Path
import pickle
from joblib import Parallel, delayed
from itertools import product
from smount_predictors.src.SeamountHelp import PipelinePredictor
from smount_predictors import SeamountHelp
import numpy as np
import pandas as pd
import simplekml
import xarray as xr
from interfaces_exclude.exclude_interface import exclude_interface

os.chdir('data')

os.system(' gmt grdmath dist2coast.grd 20 GE = coast_mask.grd')
os.system(' gmt grdmath depth_mask.grd coast_mask.grd MUL = swot_mask.grd')
os.system(' gmt grdmath swot_mask.grd vgg_swot.grd MUL = swot_landmask.grd')

os.chdir('..')

depth_grd = SeamountHelp.readAndFilterGRD("data/median_depth.gr")

longitude_pairs = []
for lon in range(-180, 180, 5):
    longitude_pairs.append((lon, lon + 5))

latitude_pairs = []
for lat in range(-80, 80, 5):
    latitude_pairs.append((lat, lat + 5))

latlons = list(product(longitude_pairs, latitude_pairs))

model = pickle.load(open('out/cluster_tuned_model.pkl', 'rb'))
data_p = Path('data/swot_landmask.grd')


def recluster(model: PipelinePredictor, predictions: pd.DataFrame):
    """
    Identify non circular shapes in the data and filters it.

    The vgg is filtered to keep only the uppter 0.7 quartile of the data
    """

    def filter_clust_size(data: pd.DataFrame):
        def circle_ratio(data: pd.DataFrame):
            if abs(data['lon'].max() - data['lon'].min()) == 0:
                return 0
            if data.loc[:, 'cluster'].mean() == -1:
                return 1
            circle = abs(data['lat'].max() - data['lat'].min()) / abs(data['lon'].max() - data['lon'].min())
            mass = (abs(data['lat'].max() - data['lat'].min()) * abs(data['lon'].max() - data['lon'].min())) / data.shape[0]
            if circle == 0:
                return 0
            mass = 4 * mass / (circle * np.pi)  # Scale mass to be 1 if it is the correct mass of a circle
            return circle * mass
        circle_range = data
        divs = circle_range.groupby('cluster').apply(circle_ratio)
        divs = divs[(divs > np.mean(divs) - np.std(divs)) & (divs < np.mean(divs) + np.std(divs))]
        circle_range = circle_range.loc[(~circle_range['cluster'].isin(divs.index)) & (circle_range['cluster'] != -1)]
        return circle_range

    def norm_z(data: pd.DataFrame):
        data['norm_z'] = (data['z'] - data['z'].min()) / (data['z'].max() - data['z'].min())
        return data
    predictions = predictions.reset_index()
    clust_filt = predictions.copy()

    clust_filt = filter_clust_size(clust_filt)
    clust_filt = norm_z(clust_filt)
    recluster_index = (clust_filt['cluster'] != -1) & (clust_filt['norm_z'] > 0.7)
    clust_pred = clust_filt.loc[recluster_index]

    model.clusterer.fit_predict(clust_pred[['lon', 'lat']])
    clust_filt.loc[recluster_index, 'cluster'] = model.clusterer.labels_ + predictions['cluster'].max() + 1
    clust_filt.loc[~recluster_index, 'cluster'] = -1
    clust_filt.set_index(['lon', 'lat'], inplace=True)
    predictions.set_index(['lon', 'lat'], inplace=True)
    predictions.loc[clust_filt.index, 'cluster'] = clust_filt.loc[:, 'cluster']
    return predictions.reset_index()


def scale_input(depth_p, data: xr.xarray):
    """Scale the input data to the seafloor depth for uniform prediciton."""
    zone = (data['lat'].min(), data['lat'].max(), data['lon'].min(), data['lon'].max())
    depth_zone = SeamountHelp.readAndFilterGRD(depth_p, zone[:1], zone[2:])
    data['z'] = data['z'] * depth_zone['z'].mean()
    return data


def predict_zone(zone):
    """Run the model on the input zone, calling all nessesary functions."""
    lon = zone[0]
    lat = zone[1]
    data = SeamountHelp.readAndFilterGRD(data_p, lat, lon)
    data = exclude_interface(data, 'interfaces_exclude/vector_feats.xy', threshold=20)
    data = scale_input(depth_grd, data)
    data = data.to_dataframe().reset_index()
    if np.all(data['z'] == 0):
        data['class'] = 0
        data['cluster'] = -1
        return data
    zone_pred = model.predict(data[['lon', 'lat', 'z']])
    return zone_pred


predictions = Parallel(n_jobs=-3)(delayed(predict_zone)(zone) for zone in latlons)
nulls = np.zeros(len(predictions), dtype=bool)
for idx, df in enumerate(predictions):  # ensure non-overlapping cluster numbers
    assert isinstance(df, pd.DataFrame)  # assertion for code linter typing features
    nulls[idx] = np.all(df['z'].values == 0)
    df.loc[df['cluster'] != -1, 'cluster'] = df.loc[df['cluster'] != -1, 'cluster'] + ((idx ** 2) * len(np.unique(df['cluster'])))
assert not np.all(nulls)
predictions = pd.concat(predictions)
predictions = recluster(model, predictions)
predictions['lon'] = np.degrees(predictions['lon'])
predictions['lat'] = np.degrees(predictions['lat'])
global_predictions = xr.Dataset.from_dataframe(predictions).set_coords(['lon', 'lat']).drop('index')
global_predictions.to_netcdf('out/global_predictions.nc')


mounts = predictions.groupby('cluster').mean().reset_index()
kml = simplekml.Kml()
for i, row in mounts.iterrows():
    kml.newpoint(name=f'{row.cluster}', coords=[(row.lon, row.lat, row.z)])
kml.save('out/global_mounts.kml')

os.system(' open out/global_mounts.kml')
