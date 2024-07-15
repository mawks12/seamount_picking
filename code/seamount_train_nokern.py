
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from bs4 import BeautifulSoup
from smount_predictors import SeamountScorer, SeamountTransformer, SeamountHelp, SeamountCVSplitter
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
import plotly.express as px
from sklearn.model_selection import GridSearchCV, train_test_split


seamount_centers = SeamountHelp.read_seamount_centers(Path('data/seamount_training_zone.kml'))[['lat', 'lon']].to_numpy()


pipe = Pipeline([
    ('trans', SeamountTransformer()),
    ('predictor', SVC(kernel='linear'))
])

param_grid = {
    'predictor__C': np.logspace(-10, 5, 10),
}

scorer = SeamountScorer(seamount_centers)

grid = GridSearchCV(pipe, param_grid, cv=SeamountCVSplitter(5), n_jobs=-1, error_score='raise', verbose=2)


points = SeamountHelp.readKMLbounds(Path('data/seamount_training_zone.kml'))
data = SeamountHelp.readAndFilterGRD(Path('data') / 'vgg_swot.grd', points[:2], points[2:])


y = SeamountHelp.readAndFilterGRD(Path('data') / 'vgg_seamounts_labled.nc')
X = data.to_dataframe().reset_index().merge(y.to_dataframe().reset_index(), on=['lat', 'lon'], how='left')
X_test = X[X['lon'] < -112.5583]
X_train = X[X['lon'] >= -112.5583]
y_train = X_train['Labels'].to_numpy()
y_test = X_test['Labels'].to_numpy()
X_train = X_train.drop(columns=['Labels'])[['lat', 'lon', 'z_x']].to_numpy()
X_test = X_test.drop(columns=['Labels'])[['lat', 'lon', 'z_x']].to_numpy()


grid.fit(X_train, y_train)


print((grid.best_score_, grid.best_params_))


print(grid.score(X_test, y_test))
