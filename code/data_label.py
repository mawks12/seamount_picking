import time
from pathlib import Path
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchgeo.samplers import RandomGeoSampler
from torchgeo.datasets import GeoDataset
from sklearn.neighbors import BallTree
import pandas as pd
from CNN import CNN, train, evaluate
from smount_predictors import SeamountHelp 

torch.manual_seed(144)
device = torch.device('mps')

data_path = Path('data/vgg_swot.grd')
data = SeamountHelp.readAndFilterGRD(data_path, lon_range=(60, 180))

labels = pd.read_csv('data/mount_heights.csv')
labels['radius'] = (labels['model_height'] / 1000) * 7.278

data = data.to_dataframe().reset_index()
start = time.time()
labeled = xr.Dataset.from_dataframe(SeamountHelp.seamount_radial_match(vgg=data, seamounts=labels).set_index(['lon', 'lat']))
print(time.time() - start)
labeled.to_netcdf('data/labled_pacific.nc')
