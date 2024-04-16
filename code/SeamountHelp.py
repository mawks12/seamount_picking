"""
Basic helper functions for common tasks
in seamount training and testing. Includes
File handling optimization and other basic task
specific functions
"""

import numpy as np
import plotly.express as px
from plotly.graph_objs._figure import Figure
from DBSCANModel import DBSCANModel
from sklearn.cluster import HDBSCAN

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
    """
    out = []
    for line in io:
        line = line.split()
        if bounds[2] <= float(line[0]) and bounds[3] >= float(line[0]):  # Check Lat bounds
            if bounds[0] <= float(line[1]) and bounds[1] >= float(line[1]):  # Check Lon bounds
                out.append([float(i) for i in line])
        else:
            continue
    return np.array(out)

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
    data = data[data[:, 1] >= bounds[2]]
    data = data[data[:, 1] <= bounds[3]]
    data = data[data[:, 0] >= bounds[0]]
    data = data[data[:, 0] <= bounds[1]]
    return data

def plotData(data, colarval="Intensity", op=1.0) -> Figure:
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
    fig.update_xaxes(
    scaleanchor="y",
    scaleratio=1,
  )
    return fig

def testNewZone(bounds, data, params=(0.32052631578947366, 13)):
    """
    Tests a new zone for seamounts using the DBSCAN algorithm
    
    Parameters
    ----------
    bounds: tuple
        area being filtered for of the form (minlat, maxlat, minlon, maxlon)
    data: np.ndarray
        data to be tested
    params: tuple
        parameters for the DBSCAN algorithm, defaults to best found params
        found in the DBSCAN training notebook
    Returns
    ----------
    labels: np.ndarray
        set of labeled data points
    score: float
        score of the zone
    """
    data = filterData(data, bounds)
    model = DBSCANModel(data, params)
    labels = model.getClusters().to_numpy()
    score = model.scoreTestData(data)
    return labels, score

def plotTestZone(bounds, data, params=(0.32052631578947366, 13)):
    """
    Tests a new zone for seamounts using the DBSCAN algorithm
    and plots the results
    
    Parameters
    ----------
    bounds: tuple
        area being filtered for of the form (minlat, maxlat, minlon, maxlon)
    data: np.ndarray
        data to be tested
    params: tuple
        parameters for the DBSCAN algorithm, defaults to best found params
        found in the DBSCAN training notebook
    """
    data = filterData(data, bounds)
    model = DBSCANModel(data, params)
    labels = model.getClusters()
    fig = plotData(labels, "Cluster")
    return fig

def divClusters(data):
    """
    Divides a set of data into clusters
    
    Parameters
    ----------
    data: np.ndarray
        data to be divided
    Returns
    ----------
    d_out: np.ndarray
        array of data with cluster labels
    centers: np.ndarray
        array of cluster centroids
    """
    clusterer = HDBSCAN(min_cluster_size=2, store_centers='centroid')
    clusterer.fit(data[:, :2])
    labels = clusterer.labels_
    d_out = np.insert(data, 2, labels, axis=1)
    centers = clusterer.centroids_
    outliers = d_out[d_out[:, 2] == -1]
    centers = np.append(centers, outliers[:, :2], axis=0)
    return d_out, centers[:, ::-1]
    