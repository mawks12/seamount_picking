import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from smount_predictors import SeamountTransformer, SeamountHelp, SeamountCVSplitter
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


pipe = Pipeline([
    ('trans', SeamountTransformer()),
    ('predictor', SVC(kernel='rbf', class_weight='balanced'))
])

param_grid = {
    'predictor__C': np.logspace(0, 5, 10),
    'predictor__gamma': ['scale', 'auto'],

}


grid = GridSearchCV(pipe, param_grid, cv=SeamountCVSplitter(5), scoring='recall', n_jobs=-1, error_score='raise', verbose=3)


points = SeamountHelp.readKMLbounds(Path('data/seamount_training_zone.kml'))
data = SeamountHelp.readAndFilterGRD(Path('data') / 'vgg_swot.grd', points[:2], points[2:])

y_cents = pd.read_csv(Path('data') / 'all.xyhrdnc', sep=' ', header=None, names=['lat', 'lon', 'height', 'radius', 'depth', 'name', 'catalouge'])

X = SeamountHelp.seamount_radial_match(data.to_dataframe().reset_index(), y_cents)
split = SeamountCVSplitter(10)
X_traini, X_testi = next(split.split(X, None))
X_train = X.iloc[X_traini]
X_test = X.iloc[X_testi]
y_train = X_train['Labels'].to_numpy()
y_test = X_test['Labels'].to_numpy()
X_train = X_train.drop(columns=['Labels'])[['lat', 'lon', 'z_x']].to_numpy()
X_test = X_test.drop(columns=['Labels'])[['lat', 'lon', 'z_x']].to_numpy()


grid.fit(X_train, y_train)


print((grid.best_score_, grid.best_params_))


print(grid.score(X_test, y_test))

with open(Path('out') / 'remote_testing.pkl', 'wb') as fout:
    pickle.dump(grid, fout)

with open(Path('out') / 'remote_testing.txt', 'w') as f2out:
    f2out.write(
        f'best_score: {grid.best_score_} \n' +
        f'best_params: {grid.best_params_}' + 
        f'training score: {grid.score(X_test, y_test)}'
        )
