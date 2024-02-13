"""
Supporting code for DBSCAN algorithm
"""
from math import atan2
from itertools import product
from sklearn.cluster import DBSCAN
import scipy.spatial as sps
import pandas as pd
import numpy as np

class DBSCANSupport:
    """
    Class containing tester functions for DBSCAN algorithm
    """
    MARGIN = 0.002  # percentage margin allowed to be considered a seamount cluster
    RADIUS = 6371  # radius of the earth in km


    def __init__(self, test_data, fast=False, test_zone=(-90, 90, -180, 180), sheet: str="new mask") -> None:
        """
        Initializes the DBSCANSupport class
        Parameters
        ----------
        test_data : str
            Path to the test data
        fast : bool
            Determines which distance function to use; default is False
            False will use more acruate haversine distance, True will use
            faster pythagorean distance
        test_zone : array-like
            Area of that the algorithm is being trained on in the form
            [min_lat, max_lat, min_lon, max_lon]; default is the whole world
        sheet : str
            Name of the sheet in the excel file to read from
        """
        self.seamounts = pd.read_excel(test_data, sheet_name=sheet)
        self.seamounts = self.seamounts.drop(columns=["VGG Height", "base_depth", "-",
                                            "Name", "Charted", "surface_depth"])  # drop unneeded columns
        self.seamounts = self.seamounts[(self.seamounts["Latitude"] >= test_zone[0]) &  # filter points out of test zone
                                        (self.seamounts["Latitude"] <= test_zone[1]) &
                                        (self.seamounts["Longitude"] >= test_zone[2]) &
                                        (self.seamounts["Longitude"] <= test_zone[3])]
        self.seamounts = self.seamounts[['Latitude', 'Longitude', 'Radius']]
        self.num_seamounts = self.seamounts.shape[0]
        self.__points = self.seamounts.to_numpy()  # get points
        self.seamount_dict = dict(zip(zip(self.__points[:, 0], self.__points[:, 1]), \
                                      self.__points[:, 2]))
        # dictionary of true seamounts and radii for faster distance checking
        self.p_neighbors = sps.KDTree(self.__points[:, :2])
        self.global_points_set = set(map(tuple, self.__points))  # set of true seamounts
        self.distance = DBSCANSupport._haversine if not fast else DBSCANSupport._pythagorean  # distance function to use

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculates the haversine distance between two points
        Parameters
        ----------
        lat1 : float
            Latitude of the first point
        lon1 : float
            Longitude of the first point
        lat2 : float
            Latitude of the second point
        lon2 : float
            Longitude of the second point
        Returns
        -------
        float
            Haversine distance between the two points in km
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * atan2(np.sqrt(a), np.sqrt(1 - a))
        return DBSCANSupport.RADIUS * c

    @staticmethod
    def _pythagorean(lat1, lon1, lat2, lon2) -> float:
        """
        Calculates the pythagorean distance between two points
        Parameters
        ----------
        lat1 : float
            Latitude of the first point
        lon1 : float
            Longitude of the first point
        lat2 : float
            Latitude of the second point
        lon2 : float
            Longitude of the second point
        Returns
        -------
        float
            Pythagorean distance between the two points in km
        """
        x = (lon2 - lon1) * np.cos((lat1 + lat2) / 2)
        y = lat2 - lat1
        return np.sqrt(x ** 2 + y ** 2) * DBSCANSupport.RADIUS

    def _trueSeamount(self, test_points) -> int:
        """
        Checks if the point is a true seamount
        Parameters
        ----------
        points : array-like
            Point to check if it is a true seamount of the form [lat, lon]
        Returns
        -------
        int
            1 if true seamount else 0
        """
        _, i = self.p_neighbors.query([test_points[0], test_points[1]])
        nearest = self.__points[i]
        radius = self.seamount_dict.get((nearest[0], nearest[1]), -1)
        if radius == -1:
            print(f"Error: {nearest[0]}, {nearest[1]} not found in seamounts")
        dist = self.distance(nearest[0], nearest[1], test_points[0], test_points[1])
        if dist < radius:
            return 1
        return -1

    def gridSearch(self, eps_vals, samp_vals, data, test, verbose=False, maxlim=False):
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
            Best score
        best_params : tuple
            Parameters that produced the best score
        best_labels : ndarray[ndarray[float, float, int]]
            Labeled data that produced the best score
        """
        best_score = -1000000
        best_params = (0.1, 1)
        best_labels = np.array([np.array([0])])
        params = list(product(eps_vals, samp_vals))  # get all possible parameter combinations
        for epsi, samp in params:  # itterate through all possible parameter combinations
            db = DBSCAN(eps=epsi, min_samples=samp)
            db.fit(data)
            labels_set = set(db.labels_)  # Convert to set to identify unique labels
            labels = db.labels_
            num_clusters = len(labels_set) - (1 if -1 in labels else 0)  # number of clusters
            if num_clusters < (2 if (test != self.outlierDeviation) else 1) or \
                num_clusters > (num_clusters + 1 if not maxlim else maxlim):
                # outlierDeviation can have fewer than 2 clusters, but not less than 1,
                # and seccond condition is to check if there is a max limit
                if verbose:
                    print(f"{epsi} and {samp} produced " + \
                          (f"{num_clusters} (too few)" if num_clusters < 2 else \
                           f'{num_clusters} (too many)') + " clusters")
                continue
            score = test(data, db.labels_)
            if verbose:
                print(f"Score for {epsi} and {samp} is {score}")
            if score > best_score:
                best_score = score
                best_params = (epsi, samp)
                best_labels = np.insert(data, 2, labels, axis=1)  # add labels to data
        return best_score, best_params, best_labels

    def autoDeviation(self, data, labels) -> float:
        """
        Calculates the deviation of output seamount
        classifications from the true seamounts using the
        fomula (true_positives - false_positives) / total_points

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
            # Identifies clusters that occur to frequently to be consitered seamounts
            if count / label_count > cluster_max_lim:
                classified = classified[classified[:, 2] != val]
        average = 0
        for i in classified:  # Itterate through model labels and check if they are in points
            average += self._trueSeamount(i)
        return average / self.num_seamounts

    def outlierDeviation(self, data, labels) -> float:
        """
        Calculates the deviation of output seamount
        classifications from the true seamounts using the
        fomula (true_positives - false_positives) / total_points
        and outliers as seamout cluster category

        Parameters
        ----------
        data : array-like
            Data that the algorithm has been fitted to
        labels : array-like
            Cluster labels for each point in data
        Returns
        -------
        deviation : float
            Deviation of output seamount classifications from true
        """
        classified = np.insert(data, 2, labels, axis=1)
        classified = classified[classified[:, 2] == -1]  # get only the outliers
        average = 0
        for i in classified:  # Itterate through model labels and check if they are in points
            average += self._trueSeamount(i)
        return average / self.num_seamounts

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
        out_data["True_Seamount"] = out_data.apply(lambda x:(self._trueSeamount((x.Latitude, x.Longitude))), axis=1)

if __name__ == "__main__":
    raise RuntimeError("DBSCANSupport is a library and should not be run as main")
