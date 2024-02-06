"""
Supporting code for debscan algorithm
"""

from itertools import product
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

class DBSCANSupport:
    """
    Class containing tester functions for DBSCAN algorithm
    """
    MARGIN = 0.002  # percentage margin allowed to be considered a seamount cluster


    def __init__(self, test_data, test_zone=(-90, 90, -180, 180), sheet="new mask") -> None:
        """
        Initializes the DBSCANSupport class
        Parameters
        ----------
        test_data : str
            Path to the test data
        test_zone : array-like
            Area of that the algorithm is being trained on in the form
            [min_lat, max_lat, min_lon, max_lon]; default is the whole world
        sheet : str
            Name of the sheet in the excel file to read from
        """
        self.seamounts = pd.read_excel(test_data, sheet_name=sheet)
        self.seamounts = self.seamounts.drop(columns=["VGG Height", "Radius", "base_depth", "-",
                                            "Name", "Charted", "surface_depth"])
        self.seamounts = self.seamounts[(self.seamounts["Latitude"] >= test_zone[0]) &
                                        (self.seamounts["Latitude"] <= test_zone[1]) &
                                        (self.seamounts["Longitude"] >= test_zone[2]) &
                                        (self.seamounts["Longitude"] <= test_zone[3])]
        self.num_seamounts = self.seamounts.shape[0]
        self.points = self.seamounts.to_numpy()


    def gridSearch(self, eps_vals, samp_vals, data, test, verbose=False):
        """
        Performs grid search on DBSCAN parameters
        Parameters
        ----------
        data : array-like
            Data to cluster
        eps_vals : list(float)
            List of epsilon values to search over
        samp_vals : list(int)
            List of min_samples values to search over
        test : function
            Function to test clustering
        Returns
        -------
        best_score : float
            Best silhouette score
        best_params : tuple
            Parameters that produced the best score
        best_labels : ndarray[ndarray[float, float, int]]
            Labeled data that produced the best score
        """
        best_score = -1
        best_params = None
        best_labels = None
        params = list(product(eps_vals, samp_vals))
        for epsi, samp in params:
            db = DBSCAN(eps=epsi, min_samples=samp)
            db.fit(data)
            labels_set = set(db.labels_)
            labels = db.labels_
            num_clusters = len(labels_set) - (1 if -1 in labels else 0)
            if num_clusters < 2:
                if verbose:
                    print(f"{epsi} and {samp} produced too few clusters")
                continue
            score = test(data, db.labels_)
            if verbose:
                print(f"Score for {epsi} and {samp} is {score}")
            if score > best_score:
                best_score = score
                best_params = (epsi, samp)
                best_labels = np.insert(data, 2, labels, axis=1)
        return best_score, best_params, best_labels

    def seamountDeviation(self, data, labels) -> float:
        """
        Calculates the deviation of output seamount
        classifications from the true seamounts

        Parameters
        ----------
        data : array-like
            Data that the algorithm has been fitted to
        labels : array-like
            Cluster labels for each point in data
        test_zone : array-like
            Area of that the algorithm is being trained on in the form
            [min_lat, max_lat, min_lon, max_lon]    
        Returns
        -------
        deviation : float
            Deviation of output seamount classifications from true
        """
        label_count = np.int64((len(labels) - (1 if -1 in labels else 0)))  # number of clusters
        classified = np.insert(data, 2, labels, axis=1)  # add labels to data
        precent_true = self.num_seamounts / classified.shape[0]  # precent of data that is actualy seamounts
        value, frequency = np.unique(labels, return_counts=True)
        value_counts = np.vstack((value, frequency)).T  # create frequency table of labels
        cluster_max_lim = precent_true + DBSCANSupport.MARGIN  # max relitive cluster size to be considered a seamount
        for val, count in value_counts:
            # Identifies clustered that occur to frequently to be consitered seamounts
            if count / label_count > cluster_max_lim:
                classified = classified[classified[:, 2] != val]
        model_labels = np.array([[round(i[0], 2), \
                                  round(i[1], 2)] for i in classified])  # get lat long of each model labeled seamount
        points_set = set(map(tuple, map(lambda x: (round(x[0], 2), round(x[1], 2)), self.points)))
        model_labels = set(map(tuple, model_labels))
        last_len = len(points_set)  # number of points that start in points
        false_positives = 0
        true_positives = 0
        for i in model_labels:  # Itterate through model labels and check if they are in points
            points_set.add(tuple(i))
            if len(points_set) > last_len:  # If the point was not in points then it is a false positive
                last_len = len(points_set)
                false_positives += 1
            else:
                true_positives += 1
        return (true_positives - false_positives) / len(self.points)

    def _getTruePair(self, test_point):
        """
        checks if the test point is a true seamount
        Parameters
        ----------
        test_point : array-like
            Point to check if it is a true seamount
        Returns
        -------
        int
            1 if true seamount else 0
        """
        if (round(test_point[0], 2), round(test_point[1], 2)) in self.points:
            return 1
        else:
            return 0

    def matchPoints(self, out_data) -> None:
        """
        adds values to indicate if the point is a true seamount
        Parameters
        ----------
        out_data : pd.DataFrame
            Data to add values to
        Returns
        -------
        None
        """
        out_data["True Seamount"] = out_data.apply(lambda x:(self._getTruePair((x.Longitude, x.Latitude))), axis=1)
