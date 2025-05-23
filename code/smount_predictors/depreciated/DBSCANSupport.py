"""
Supporting code for DBSCAN algorithm
"""

from itertools import product
from sklearn.cluster import DBSCAN
import numpy as np
from ._SeamountSupport import _SeamountSupport
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

class DBSCANSupport(_SeamountSupport):
    """
    Class containing tester functions for DBSCAN algorithm
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
        self.end_params = None  # values of the best parameters
        self.test = False

    def gridSearch(self, eps_vals, samp_vals, verbose=False, maxlim=False):
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
        if self.training_data is None or self.unlabled_data is None:
            raise AttributeError("Training data has not been added to the class yet")
        best_score = 100000000
        params = list(product(eps_vals, samp_vals))  # get all possible parameter combinations
        classifier = self.__autoFilter
        for epsi, samp in params:  # itterate through all possible parameter combinations
            classifier = self.__autoFilter
            db = DBSCAN(eps=epsi, min_samples=samp)
            db.fit(self.unlabled_data)
            labels_set = set(db.labels_)  # Convert to set to identify unique labels
            labels = db.labels_
            num_clusters = (len(labels_set) - (1 if -1 in labels else 0)) \
                if classifier is not DBSCANSupport.__outlierFilter else len(labels_set)  # number of clusters
            max_clusters = (num_clusters + 1 if not maxlim else maxlim)
            min_clusters = 1
            if num_clusters < min_clusters or num_clusters > max_clusters:
                if verbose:
                    print(f"{epsi} and {samp} produced " + \
                          (f"too few {labels_set}" if num_clusters < min_clusters else \
                           f'too many {labels_set}') + " clusters")
                continue
            if len(labels_set) == 2 and -1 in labels_set:
                if verbose:
                    print(f"using outlier deviation on {epsi} and {samp}")
                    out = True
                classifier = DBSCANSupport.__outlierFilter
            else:
                out = False
            score = self.deviation(self.unlabled_data, labels, classifier)
            if verbose:
                print(f"Score for {epsi} and {samp} is {score} with {len(labels_set)} clusters")
            if score < best_score and score != 0:
                best_score = score
                self.end_params = (epsi, samp)
                best_labels = labels  # add labels to data
                self.test = out
        if self.end_params is None:
            raise AttributeError("No valid parameters found")
        if verbose:
            print(f"Best score: {best_score} with parameters {self.end_params}")
        best_labels = np.insert(  # return non scaled data
            self.datascaler.inverse_transform(self.training_data), 2, best_labels, axis=1)  # type: ignore
        return best_score, self.end_params, best_labels, self.test

    def deviation(self, data, labels, classifier):
        """
        Calculates the deviation of output seamount
        classifications from the true seamounts using the
        formula (false postivies + false negetives) / total true positives
        and the filter variable to determine how to assign clusters
        Parameters
        ----------
        data : array-like
            Data that the algorithm has been fitted to
        labels : array-like
            Cluster labels for each point in data
        filter : function
            Function to filter clusters
        classifier : function
            Function to classify clusters
        valid : bool | np.ndarray
            If false, will default to using the pre assinged
            validation data, else will use the data provided
        """
        classified = classifier(data, labels)
        vals = np.ndarray(shape=(len(classified), 3))
        assert self.seamount_points is not None
        if len(classified) == 0:
            raise ValueError("Classifier returned no valid clusters")
        i = 0
        for val in classified:  # Iterate through model labels and check if they are in points
            mount = self._trueSeamount(val[0:2])
            if mount == 0:
                mount = 1
            vals[i][0] = mount
            vals[i][1] = 1
            #nearest, radius = self.getNear(val[:2])
            #vals[i][2] = _SeamountSupport.pDist(radius, nearest, classified[i])
            i += 1
        return metrics.log_loss(vals[:, 0], classified[:, 2], labels=[-1, 1])

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
        nonval = stats.mode(labels)[0]
        classified = np.insert(data, 2, labels, axis=1)
        classified[classified[:, 2] == nonval][:, 2] = -2
        classified[classified[:, 2] != nonval][:, 2] = 1
        classified[classified[:, 2] == -2][:, 2] = -1
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
        classified[classified[:, 2] != -1][:, 2] = -2
        classified[classified[:, 2] == -1][:, 2] = 1
        classified[classified[:, 2] == -2][:, 2] = -1
        return classified

    def testData(self, test_data) -> np.ndarray:
        """
        Tests model on new data
        Parameters
        ----------
        test_data : array-like
            Data to score
        Returns  TODO: Update docs
        -------
        score : float
            Score of the test data
        """
        if self.end_params is None:
            raise AttributeError("Training has not been done")
        db = DBSCAN(eps=self.end_params[0], min_samples=self.end_params[1])
        scalar = StandardScaler()
        test_data = scalar.fit_transform(test_data)
        db.fit(test_data)
        return np.insert(scalar.inverse_transform(test_data), 2, db.labels_, axis=1)  # type: ignore

    def getSeamountPoints(self):
        """
        Returns the points that got catagorized as seamounts
        """
        if self.end_params is None:
            raise AttributeError("Testing has not been done")
        if self.test:
            return (-1,)
        db = DBSCAN(eps=self.end_params[0], min_samples=self.end_params[1]).fit(self.unlabled_data)  # type: ignore
        label_count = np.int64((len(db.labels_) - (1 if -1 in db.labels_ else 0)))  # number of clusters
        precent_true = self.seamount_points / self.training_data.shape[0]  # type: ignore
        # precent of data that is actualy seamounts
        value, frequency = np.unique(db.labels_, return_counts=True)
        value_counts = np.vstack((value, frequency)).T  # create frequency table of labels
        cluster_max_lim = precent_true + DBSCANSupport.MARGIN  # max relitive cluster size to be considered a seamount
        vals = []
        for val, count in value_counts:
            # Identifies clusters that occur to frequently to be consitered seamounts
            if count / label_count < cluster_max_lim:
                vals.append(val)
        return tuple(vals)

    def makePredict(self, data: np.ndarray):
        ...  # TODO: Implement makePredict

if __name__ == "__main__":
    raise RuntimeError("DBSCANSupport is a library and should not be run as main")
