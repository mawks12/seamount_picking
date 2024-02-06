import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from DBSCANSupport import *

src_dir = '/Users/seamount_picking/data/'  # Update to local machine
grav = pd.read_csv(src_dir + 'test_grav.csv')#.drop(columns=["old_ind"])
X = grav[['Longitude', 'Latitude', 'Intensity']].to_numpy()

scale_x = X
test_eps = np.linspace(0.01, 1, 50)
test_samp = np.arange(15, 50)
scnsprt = DBSCANSupport("/Users/seamount_picking/data/sample_mask.txt.xlsx")

score, params, data_out  = scnsprt.gridSearch(test_eps, test_samp, scale_x, scnsprt.seamountDeviation, verbose=True)

data_out = pd.DataFrame({'Longitude': data_out[:, 0], 'Latitude': data_out[:, 1], 'Intensity': data_out[:, 2], 'score': data_out[:, 3]})

data_out.to_csv(src_dir + 'best_test_grav_nomaxclust_clusters.csv', index=False)
