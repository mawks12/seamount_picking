from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.neighbors import BallTree

batched_dir = Path('data/mount_samples')

mount_path = Path('data/mount_heights.csv')
mounts = pd.read_csv(mount_path)
mounts['radius'] = mounts['model_height'] * 7.278

data_path = Path('data/labled_pacific.nc')
vgg_labels = xr.open_dataset(data_path)

flat_data = vgg_labels.to_dataframe().reset_index()
tree = BallTree(flat_data[['lon', 'lat']].to_numpy(), leaf_size=2)
for seamount in mounts.itertuples():
    _, center_ind = tree.query(np.radians([[seamount.lon, seamount.lat]]), k=1)
    center_ind = center_ind[0][0]
    center = flat_data[['lon', 'lat']].iloc[center_ind].values.flatten()
    print(center)
    padding = seamount.radius * 3 + 1000
    lat_len = padding / np.degrees(6378137)
    lat_bounds = (center[1] - lat_len, center[1] - lat_len)
    lon_vals = np.degrees(6378137 * np.cos(np.radians(center[1])))
    lon_bounds = (center[0] - lon_vals, center[0] + lon_vals)
    mount_loc = vgg_labels.query(lat=f'lat > {lat_bounds[0]} & lat < {lat_bounds[1]}', lon=f'lon > {lon_bounds[0]} & lon < {lon_bounds[1]}')
    mount_loc.to_netcdf(batched_dir / f'sample_{seamount.name}.nc')
