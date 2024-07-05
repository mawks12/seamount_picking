"""
Basic helper functions for common tasks
in seamount training and testing. Includes
File handling optimization and other basic task
specific functions
"""

from pathlib import Path
from typing import Generator
import numpy as np
import plotly.express as px
from plotly.graph_objs._figure import Figure
import xarray as xr
from bs4 import BeautifulSoup
import pandas as pd

def readCroppedxyz(io,  bounds: tuple[float, float, float, float]) -> np.ndarray:
    """
    Reads a large xyz file line by line and filters for area within
    defined bounds to reduce memory load

    Parameters
    ----------
    io: _io.TextIOWrapper
        text wrapper object of file being read
    bounds: tuple
        area being filtered for of the form (minlat, maxlat, minlon, maxlon)
    Returns
    -------
    np.ndarray
        2d array where each sub array is of the form [Lon, Lat, Zval]
    """
    out = []
    for line in io:
        line = line.split()
        if bounds[2] <= float(line[0]) and bounds[3] >= float(line[0]):  # Check Lon bounds
            if bounds[0] <= float(line[1]) and bounds[1] >= float(line[1]):  # Check Lat bounds
                out.append([float(i) for i in line])
        else:
            continue
    return np.array(out)[:, [1, 0, 2]]

def filterData(data: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    """
    Filters data for a specific area

    Parameters
    ----------
    data: np.ndarray
        data to be filtered, of the form lat, lon
    bounds: tuple
        area being filtered for of the form (minlat, maxlat, minlon, maxlon)
    Returns
    ----------
    filtered: np.ndarray
        filtered data
    """
    data = data[data[:, 0] >= bounds[0]]
    data = data[data[:, 0] <= bounds[1]]
    data = data[data[:, 1] >= bounds[2]]
    data = data[data[:, 1] <= bounds[3]]
    return data

def plotData(data, colarval="Intensity", op=1) -> Figure:
    """
    generates a plot of Lat Lon Intensity data
    Parameters
    ----------
    data: pd.Dataframe
        dataframe of data
    Returns
    ----------
    fig: Figure
        plot of data
    """
    fig = px.scatter(data, x="Longitude", y="Latitude", color=colarval, opacity=op)
    fig = px.scatter(data, x="Longitude", y="Latitude", color=colarval)
    fig = px.scatter(data, x="Longitude", y="Latitude", color=colarval)
    fig.update_xaxes(
    scaleanchor="y",
    scaleratio=1,
  )
    return fig

def readAndFilterGRD(file_path: Path, lat_range: tuple[float, float], lon_range: tuple[float, float]) -> xr.Dataset:
    """
    Reads a grd file into an xarray dataset and filters it based on specified lat and lon ranges.

    Parameters
    ----------
    file_path: str
        Path to the grd file.
    lat_range: tuple
        Range of latitudes to filter for, in the form (min_lat, max_lat).
    lon_range: tuple
        Range of longitudes to filter for, in the form (min_lon, max_lon).

    Returns
    -------
    xr.Dataset
        Filtered xarray dataset containing the data from the grd file within the specified lat and lon ranges.
    """
    dataset = xr.open_dataset(file_path, engine='netcdf4')
    filtered_dataset = dataset.sel(
        lat=slice(lat_range[0], lat_range[1]), lon=slice(lon_range[0], lon_range[1])
        )
    return filtered_dataset

def readKMLbounds(file: Path) -> tuple[float, float, float, float]:
    """
    Reads a KML file and extracts the lat and lon bounds
    """
    # Load your KML file
    with open(file, "r") as fin:
        kml_content = fin.read()

    soup = BeautifulSoup(kml_content, 'xml')

    # Find all Placemark tags
    placemarks = soup.find_all('Placemark')

    # Initialize a list to hold the coordinates
    coordinates_list = []

    for placemark in placemarks:
        # Check if the name of the Placemark starts with 'q' followed by a digit
        if placemark.find('name') and placemark.find('name').text.startswith('q') \
            and placemark.find('name').text[1:].isdigit():
            # Extract the coordinates
            coordinates = placemark.find('coordinates')
            if coordinates:
                coordinates_list.append(coordinates.text.strip())
    minlat = 91
    maxlat = -91
    minlon = 181
    maxlon = -181
    for pair in coordinates_list:
        lon, lat, _ = pair.split(',')
        lat = float(lat)
        lon = float(lon)
        if lat < minlat:
            minlat = lat
        if lat > maxlat:
            maxlat = lat
        if lon < minlon:
            minlon = lon
        if lon > maxlon:
            maxlon = lon
    return (minlat, maxlat, minlon, maxlon)

def show_convolutions(results: dict) -> Generator[Figure, None, None]:
    """
    Generates plots for all of the convolutional layers in the results dictionary.
    """
    keyvals = sorted(results.items(), key=lambda x: int(x[0].split('_')[1]) + float(x[0].split('_')[2]) / 100)
    for key, val in keyvals:
        fig = plot_xarr(val, name=key)
        yield fig

def plot_xarr(data: xr.Dataset, name: str = 'Untitled') -> Figure:
    """
    Plots an xarray dataset
    """
    fig = px.scatter(data.to_dataframe().reset_index(), x='lon', y='lat', color='z', title=name)
    fig.update_layout(
        width=800,
        height=800,
        xaxis=dict(type='linear', autorange=True),  # Adjust x-axis properties
        yaxis=dict(type='linear', autorange=True),  # Adjust y-axis properties
    )
    return fig

def adjust_lon(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Adjusts the longitude values in the dataset to account for
    the distortion farther from the equator.
    """
    data['lon'] = data['lon'] / np.cos(np.radians(data['lat']))
    return data

def inverse_adjust_lon(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Inverse of adjust_lon function.
    """
    data['lon'] = data['lon'] * np.cos(np.radians(data['lat']))
    return data

def normalize_kernel(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalizes the kernel values to sum to 1.
    """
    min_val = ds['z'].values.min()
    ds['z'].values = ds['z'].values + abs(min_val)
    ds['z'].values = ds['z'].values / ds['z'].values.sum()
    return ds

def xar_from_numpy(data: np.ndarray) -> xr.Dataset:
    """
    Converts a numpy array to an xarray dataset.
    """
    df = pd.DataFrame({
        'lon': data[:, 0],
        'lat': data[:, 1],
        'z': data[:, 2]
    })
    ds = xr.Dataset.from_dataframe(df.set_index(['lon', 'lat']))
    return ds
