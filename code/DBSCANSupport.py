"""
Supporting code for DBSCAN algorithm
"""

from itertools import product
from sklearn.cluster import DBSCAN
import scipy.spatial as sps
import numpy as np
from _SeamountSupport import _SeamountSupport


class DBSCANSupport(_SeamountSupport):
    """
    Class containing tester functions for DBSCAN algorithm
    Converts lat long to UTM coordinates for euclidean distance 
    algorithms, and but currently only works with one UTM zone
    at a time # TODO: Update to work with more
    """
    MARGIN = 0.002  # percentage margin allowed to be considered a seamount cluster
    RADIUS = 6371  # radius of the earth in km


    def __init__(self, test_data, train_zone=(-90, 90, -180, 180), sheet: str="new mask") -> None:
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
        super().__init__(test_data, train_zone, sheet)
        self.zone_number = 15  # TODO: remove hardcoding
        self.end_params = (0.1, 1)  # default parameters for the end of the grid search

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
        best_score = -1000000.0
        best_params = (0.1, 1)
        best_labels = np.array([np.array([0])])
        params = list(product(eps_vals, samp_vals))  # get all possible parameter combinations
        if test == "auto":
            classifier = self.__autoFilter
        else:
            classifier = DBSCANSupport.__outlierFilter
        for epsi, samp in params:  # itterate through all possible parameter combinations
            db = DBSCAN(eps=epsi, min_samples=samp)
            db.fit(data)
            labels_set = set(db.labels_)  # Convert to set to identify unique labels
            labels = db.labels_
            num_clusters = len(labels_set) - (1 if -1 in labels else 0)  # number of clusters
            max_clusters = (num_clusters + 1 if not maxlim else maxlim)
            min_clusters = (2 if not (test is self.__autoFilter) else 1)
            if num_clusters < min_clusters or num_clusters > max_clusters:
                # outlierDeviation can have fewer than 2 clusters, but not less than 1,
                # and seccond condition is to check if there is a max limit
                if verbose:
                    print(f"{epsi} and {samp} produced " + \
                          (f"{num_clusters} (too few)" if num_clusters < 2 else \
                           f'{num_clusters} (too many)') + " clusters")
                continue
            score = self.deviation(data, db.labels_, classifier)
            if verbose:
                print(f"Score for {epsi} and {samp} is {score}")
            if score > best_score:
                best_score = score
                best_params = (epsi, samp)
                best_labels = np.insert(data, 2, labels, axis=1)  # add labels to data
        self.end_params = best_params
        return best_score, best_params, best_labels

    def deviation(self, data, labels, classifier, valid=False):
        """
        Calculates the deviation of output seamount
        classifications from the true seamounts using the
        fomula (true_positives - false_positives) / total_points
        and the filter variable to determine how to assign clusters
        Parameters
        ----------
        data : array-like
            Data that the algorithm has been fitted to
        labels : array-like
            Cluster labels for each point in data
        filter : function
            Function to filter clusters
        valid : bool | np.ndarray
            If false, will default to using the pre assinged
            validation data, else will use the data provided
        """
        if not valid:
            valid = self.p_neighbors
        else:
            __points = valid
            valid = sps.KDTree(__points[:, :2])  # type: ignore
        if classifier == "auto":
            classified = self.__autoFilter(data, labels)
        else:
            classified = DBSCANSupport.__outlierFilter(data, labels)
        average = 0
        for i in classified:  # Itterate through model labels and check if they are in points
            average += self._trueSeamount(i)
        return average / self.num_seamounts

    def __autoFilter(self, data, labels): # pylint: disable=invalid-name
        """
        Selects clusters that are small enough to be
        consitered potential seamounts
        Parameters
        ----------
        data : array-like
            Data that the algorithm has been fitted to
        labels : array-like
            Cluster labels for each point in data
        Returns
        -------
        classified : np.ndarray
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
        return classified

    @staticmethod
    def __outlierFilter(data, labels) -> np.ndarray: # pylint: disable=invalid-name
        """
        Filters out the outliers from the data
        Parameters
        ----------
        data : array-like
            Data that the algorithm has been fitted to
        labels : array-like
            Cluster labels for each point in data
        Returns
        -------
        filtered : np.ndarray
            Filtered data
        """
        classified = np.insert(data, 2, labels, axis=1)
        return classified[classified[:, 2] == -1]

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
        out_data["True_Seamount"] = out_data.apply(lambda x:(self._trueSeamount((x.Easting, x.Northing))), axis=1)

    def scoreTestData(self, data_range, path, params, test_data, *args, test="outlier") -> float:
        """
        Scores the test data
        Parameters
        ----------
        path : str
            Path to the test data
        params : tuple
            Parameters for the DBSCAN algorithm
        test_data : array-like
            Data test algorithm on
        data_range : array-like
            Range of data to score of the form [min_lat, max_lat, min_lon, max_lon]
        test : function
            Function to test clustering
        Returns
        -------
        score : float
            Score of the test data
        """
        _ = args
        valid = self._filterData(path, data_range)
        db = DBSCAN(eps=params[0], min_samples=params[1])
        db.fit(test_data)
        clissifier = self.__autoFilter if test == "auto" else DBSCANSupport.__outlierFilter
        return self.deviation(test_data, db.labels_, clissifier, valid)  # type: ignore

if __name__ == "__main__":
    raise RuntimeError("DBSCANSupport is a library and should not be run as main")
