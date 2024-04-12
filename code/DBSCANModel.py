'''
DBSCANModel.py
- This file contains the DBSCANModel class, which is a class that is used to create a DBSCAN model for clustering.
'''

from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import log_loss
import pandas as pd
from DBSCANSupport import DBSCANSupport

class DBSCANModel(DBSCANSupport):
    """
    DBSCANModel class is used to create a DBSCAN model for clustering.
    It is built on the support class used to train the model, but 
    is used to create a model for a specific set of data.
    """

    def __init__(self, data: np.ndarray, params: tuple = (0.32052631578947366, 13)):
        """
        Initializes the DBSCAN model with the data and parameters
        
        Parameters
        ----------
        data: np.ndarray
            data to be tested
        params: tuple
            parameters for the DBSCAN algorithm, defaults to best found params
        """
        self.model = DBSCAN(eps=params[0], min_samples=params[1])
        self.data = self.datascaler.fit_transform(data)
        self.model.fit(data)
        self.data = np.ndarray(self.datascaler.inverse_transform(data))  # type: ignore
        trainzone = (self.data[:, 0].min(), self.data[:, 0].max(),
                     self.data[:, 1].min(), self.data[:, 1].max()) # type: ignore
        super().__init__(Path('data') / 'sample_mask.txt.xlsx', train_zone=trainzone)
        self.params = params
        self.addTrainingData(self.data)
        self.labels = self.model.labels_

    def getClusters(self) -> pd.DataFrame:
        """
        Returns the clusters found by the DBSCAN algorithm
        
        Returns
        ----------
        clusters: pd.DataFrame
            dataframe of clusters
        """
        clusters = pd.DataFrame(self.data, columns=["Latitude", "Longitude", "Intensity"])  # type: ignore
        clusters["Cluster"] = self.labels
        return clusters

    def getClusterCount(self) -> float:
        """
        Returns the number of clusters found by the DBSCAN algorithm
        
        Returns
        ----------
        clusterCount: int
            number of clusters
        """
        cluster_count = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        return cluster_count

    def scoreTestData(self, test_data) -> float:
        """
        Scores the test data using the log loss function
        Parameters
        ----------
        test_data: np.ndarray
            data to be tested
        Returns
        ----------
        score: float
            score of the test data
        """
        return float(log_loss(self.training_data[:, 3], test_data)) # type: ignore
