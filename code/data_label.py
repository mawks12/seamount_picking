import time
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from smount_predictors import SeamountHelp 

data_path = Path('data/vgg_swot.grd')
data = SeamountHelp.readAndFilterGRD(data_path, lon_range=(60, 180))

labels = pd.read_csv('data/mount_heights.csv')
labels['radius'] = (labels['model_height'] / 1000) * 7.278

data = data.to_dataframe().reset_index()
labeled = xr.Dataset.from_dataframe(SeamountHelp.seamount_radial_match(vgg=data, seamounts=labels).set_index(['lon', 'lat']))
labeled.to_netcdf('data/labled_pacific.nc')
