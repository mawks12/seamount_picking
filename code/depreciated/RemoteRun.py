# %%
import pandas as pd
import numpy as np
from DBSCANSupport import DBSCANSupport
from LocalPath import LOCALPATH

# %%
grav = pd.read_csv(LOCALPATH + 'data/test_grav.csv').drop(columns=["old_ind"])
grav = grav[['Latitude', 'Longitude', 'Intensity']]

# %%
Z = DBSCANSupport.formatData(grav, 'Intensity')
#Z = Z[(Z[:, 2] > -20) & (Z[:, 2] < 7)]
data = Z

# %%
test_eps = np.linspace(2.3, 2.8, 20)
test_samp = np.arange(2, 7)
DBModel_test = DBSCANSupport(LOCALPATH+"data/sample_mask.txt.xlsx", train_zone=(-6, -1.5, -98, -90))
DBModel_test.addTrainingData(data)

# %%
score, params, data_out  = DBModel_test.gridSearch(test_eps, test_samp, verbose=True)

# %%
dfout = pd.DataFrame(data_out, columns=["Easting", "Northing", "Label", "Intensity"])
df_labeled = dfout[dfout['Label'] == 1]

# %%
with open(LOCALPATH + "out/best.txt", "w") as f:
    f.write("Score: " + str(score) + "\n")
    f.write("Params: " + str(params) + "\n")
DBModel_test.matchPoints().to_csv(LOCALPATH + "out/matched.csv", index=False)
dfout.to_csv(LOCALPATH + "out/labels.csv", index=False)
