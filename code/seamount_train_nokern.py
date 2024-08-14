import xarray as xr
from pathlib import Path
import pickle
from sklearn.cluster import HDBSCAN  # type: ignore
import numpy as np
from smount_predictors import SeamountScorer, SeamountTransformer, SeamountHelp, SeamountCVSplitter
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


seamount_centers = SeamountHelp.read_seamount_centers(Path('data/seamount_training_zone.kml'))[['lat', 'lon']].to_numpy()


pipe = Pipeline([
    ('trans', SeamountTransformer()),
    ('predictor', SVC(kernel='rbf', class_weight='balanced'))
])

param_grid = {
    'predictor__C': np.logspace(0, 9, 10),
}

scorer = SeamountScorer(seamount_centers)

grid = GridSearchCV(pipe, param_grid, cv=SeamountCVSplitter(5), n_jobs=-1, error_score='raise', verbose=3)


points = SeamountHelp.readKMLbounds(Path('data/seamount_training_zone.kml'))
data = xr.open_dataset('out/training_set.nc', engine='netcdf4')
X = data.to_dataframe().reset_index()

splitter = SeamountCVSplitter(5)
X_train, X_test = next(splitter.split(X[['lat', 'lon', 'z']], X['Labels']))
X_train = X.iloc[X_train]
y_train = X_train['Labels'].to_numpy()
X_train = X_train[['lat', 'lon', 'z']].to_numpy()
X_test = X.iloc[X_test]
y_test = X_test['Labels'].to_numpy()
X_test = X_test[['lat', 'lon', 'z']].to_numpy()

grid.fit(X_train, y_train)


print((grid.best_score_, grid.best_params_))


print(grid.score(X_test, y_test))


class PipelinePredictor:
    def __init__(self, model, clusterer):
        self.model = model
        self.clusterer = clusterer

    def predict(self, data):
        predictions = self.model.predict(data)
        data['class'] = predictions
        self.clusterer.fit_predict(data[['lon', 'lat', 'class']])
        data['cluster'] = self.clusterer.labels_
        return data
    
full_pipeline = PipelinePredictor(grid, HDBSCAN())
pickle.dump(full_pipeline, open('out/rbf_model.pkl', 'wb'))

with open(Path('out') / 'remote_testing.txt', 'w') as f2out:
    f2out.write(
        f'best_score: {grid.best_score_} \n' +
        f'best_params: {grid.best_params_}' + 
        f'training score: {grid.score(X_test, y_test)}'
        )
