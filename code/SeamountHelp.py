"""
Basic helper functions for common tasks
in seamount training and testing. Includes
File handling optimization and other basic task
specific functions
"""

import numpy as np
import plotly.express as px
from plotly.graph_objs._figure import Figure

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

def plotData(data, colarval="Intensity") -> Figure:
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
    fig = px.scatter(data, x="Longitude", y="Latitude", color=colarval)
    fig.update_xaxes(
    scaleanchor="y",
    scaleratio=1,
  )
    return fig
