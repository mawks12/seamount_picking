# %%
from GaussianSupport import GaussianSupport
import pandas as pd
import numpy as np
import plotly.express as px
from LocalPath import LOCALPATH
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

grav = pd.read_csv(LOCALPATH + 'data/test_curv_32.highpass.csv', names=['Longitude', 'Latitude', 'Intensity'], header=0)
grav = grav[['Longitude', 'Latitude', 'Intensity']]

# %%
Z = GaussianSupport.formatData(grav, 'Intensity')
data = Z
GausModel_test = GaussianSupport(LOCALPATH+"data/sample_mask.txt.xlsx", train_zone=(-6, -1, -98, -90))
GausModel_test.addTrainingData(data)

# %%
testing_model = GaussianProcessClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
assert isinstance(GausModel_test.training_data, np.ndarray)
in_dat = GausModel_test.training_data[:, :3]
out_dat = GausModel_test.training_data[:, 3]

# %%
scores = cross_val_score(testing_model, in_dat, out_dat, scoring='accuracy', cv=cv, n_jobs=1)
print(np.mean(scores))

# %%
#GausModel_test.fitGauss(data)

# %%



