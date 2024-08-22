import numpy as np
import xarray as xr
import os
# import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import pygmt
import sys
from collections import defaultdict
from geopy.distance import geodesic


def exclude_interface(dataset: xr.Dataset, ascii_file_path='vector_features.xy', threshold = 10):

    
    # Reads a .grd file and an ASCII file containing paths, then applies a mask to the grid data
    # based on proximity to paths.

    # Parameters:
    # grd_file (str): SWOT VGG .grd file.
    # out_file (str): VGG .grd file with interface mask applied
    # ascii_file_path (str): The path to the ASCII file containing paths.
    # to make it from kmz file: gmt kml2gmt vector_features.kml -V > vector_features.xy
    # threshold (int, optional): The distance threshold in km for masking. Default is 10 km.

    # Returns:
    # xarray.Dataset: The masked grid data.
    def interpolate_path(path, max_distance_km=5):
        interpolated_path = []
        for i in range(len(path) - 1):
            start_point = path[i]
            end_point = path[i + 1]
            distance = geodesic(start_point, end_point).km
            if distance <= max_distance_km:
                interpolated_path.append(start_point)
            else:
                num_points = int(distance // max_distance_km) + 1
                lat_step = (end_point[0] - start_point[0]) / num_points
                lon_step = (end_point[1] - start_point[1]) / num_points
                for j in range(num_points):
                    interpolated_path.append((start_point[0] + j * lat_step, start_point[1] + j * lon_step))
        interpolated_path.append(path[-1])
        return interpolated_path

    # Function to check if a path contains points within the specified area
    def path_in_studied_area(path, lon_min, lon_max, lat_min, lat_max):
        for lat, lon in path:
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                return True
        return False

    data = dataset
    # Extract the latitude and longitude data
    latitudes = data['lat'].values
    longitudes = data['lon'].values
    lon_min = longitudes.min()
    lon_max = longitudes.max()
    lat_min = latitudes.min()
    lat_max = latitudes.max()

    # Load the ASCII file containing paths
    ascii_file_path = 'vector_features.xy'
    paths = defaultdict(list)
    current_path = None
     
    with open(ascii_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('> -L'):
                parts = line.split('-D')
                if len(parts) > 1:
                    description_part = parts[1].strip('"')
                    current_path = description_part.split(' ')[0]
                else:
                    current_path = parts[0].split('"')[1]
                paths[current_path] = []  # Initialize the path list
            elif current_path is not None and line:
                lon, lat = map(float, line.split())
                paths[current_path].append((lat, lon))
                
    # Interpolate additional points on each path
    for key in paths:
        paths[key] = interpolate_path(paths[key], max_distance_km=threshold)

    # Filter paths to only include those within the specified area
    filtered_paths = {key: path for key, path in paths.items() if path_in_studied_area(path, lon_min, lon_max, lat_min, lat_max)}


    # Create a mask for setting values to NaN within 10 km of any path
    mask = np.ones_like(data['z'].values, dtype=bool)

    # loop over each point on path:
    for path in list(filtered_paths.values()):
        for sub_path in path:
            lat = sub_path[0]
            lon = sub_path[1]
            # find pixels within 10 km of the path point
            distance_y = np.abs(latitudes - lat)*111.2 # spacing in km
            distance_x = np.abs(longitudes - lon)*111.2*np.cos(latitudes.mean()/180*np.pi) # spacing in km
            distance = (np.transpose(np.tile(distance_y**2, (distance_x.shape[0],1))) + np.tile(distance_x**2, (distance_y.shape[0],1)))**0.5
            mask[distance<threshold] = True

    # Set the z values to NaN where the mask is False
    data['z'].values[mask] = 0
    return data
