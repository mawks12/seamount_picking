"""
Basic support class for the seamount detection code
"""

from abc import abstractmethod
from math import atan2
import scipy.spatial
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class _SeamountSupport:
    """
    Basic Support class for the seamount detection code
    Contains basic training, testing, and distance functions, as well
    as all of the general static methods used in the seamount
    detection code.
    """
    MARGIN = 0.002  # percentage margin allowed to be considered a seamount cluster
    RADIUS = 6371  # radius of the earth in km
    FILTERTHRSH = -0.5  # threshold for filtering out points

    def __init__(self, validation_data, train_zone=(-90, 90, -180, 180), sheet: str="new mask") -> None:
        """
        Initializes the DBSCANSupport class
        Parameters
        ----------
        validation_data : str
            Path to the data used for validation
        fast : bool
            Determines which distance function to use; default is False
            False will use more acruate haversine distance, True will use
            faster pythagorean distance
        train_zone : array-like
            Area of that the algorithm is being trained on in the form
            [min_lat, max_lat, min_lon, max_lon]; default is the whole world
        sheet : str
            Name of the sheet in the excel file to read from
        """
        self.sheet = sheet
        self.validation_path = validation_data
        self.train_zone = train_zone
        seamounts = self._filterData(self.validation_path, self.train_zone)
        self.num_seamounts = seamounts.shape[0]
        self.distance = _SeamountSupport._pythagorean  # distance function
        self.datascaler = StandardScaler()
        self.unlabled_data = None
        self.label_hash = None
        self.training_data = None

    def _trueSeamount(self, test_points):
        """
        Checks if a point is a true seamount by comparing
        it to the its hashed value
        Parameters
        ----------
        test_points : array-like
            Point to check if it is a true seamount
        Returns
        -------
        int
            1 if true seamount else -1
        """
        if self.unlabled_data is None or self.label_hash is None:
            raise AttributeError("No training data has been added")
        hashed = self.label_hash.get(tuple(test_points), 0)
        if hashed == 0:
            raise ValueError(f"{test_points} not found in training data")
        return hashed

    def addTrainingData(self, training_data: np.ndarray) -> None:
        """
        Adds training data to the class
        Parameters
        ----------
        training_data : np.ndarray
            Data to add to the class
        Returns
        -------
        None
        """
        self.training_data = training_data
        self.datascaler.fit(training_data)
        seamounts = self._filterData(self.validation_path, self.train_zone)
        __points = seamounts.to_numpy()  # get points
        seamount_dict = dict(zip(zip(__points[:, 0], __points[:, 1]), \
                                      __points[:, 2]))
        # dictionary of true seamounts and radii for faster distance checking
        p_neighbors = scipy.spatial.KDTree(__points[:, :2])
        for i in range(self.training_data.shape[0]):
            self.training_data[i][3] = _SeamountSupport._radiusMatch(
                training_data[i], p_neighbors, __points, seamount_dict)
        self.training_data = self.datascaler.transform(training_data)
        self.training_data = self.training_data[self.training_data[:, 2] > _SeamountSupport.FILTERTHRSH]  # type: ignore
        self.unlabled_data = self.training_data[:, :3]  # type: ignore
        assert isinstance(self.unlabled_data, np.ndarray)
        self.label_hash = dict(zip(map(tuple, self.unlabled_data[:, :2]), training_data[:, 3]))

    @staticmethod
    def _radiusMatch(test_points, tree, points, query) -> int:
        """
        Checks if a point is a true seamount by comparing
        it to the its hashed value
        Parameters
        ----------
        test_points : array-like
            Point to check if it is a true seamount
        tree : scipy.spatial.kdtree
            KDTree of the training data
        points : np.ndarray
            Training data
        query : dict
            Dictionary of true seamounts and radii for faster distance checking
        Returns
        -------
        int
            1 if true seamount else 0
        """
        _, i = tree.query([test_points[0], test_points[1]])
        nearest = points[i]
        radius = query.get((nearest[0], nearest[1]), -1)
        if radius == -1:
            raise ValueError(f"Error: {nearest[0]}, {nearest[1]} not found in seamounts")
        dist = _SeamountSupport._pythagorean(nearest[0], nearest[1], test_points[0], test_points[1])
        if dist < radius:
            return 1
        return -1

    def _filterData(self, path, data_range: tuple, csv=False) -> pd.DataFrame:
        """
        Filters the data to only include the testing zone
        Parameters
        ----------
        data_range : tuple
            Range of data to filter in the form (min_lat, max_lat, min_lon, max_lon)
        Returns
        -------
        pd.DataFrame
            Filtered data
        """
        if csv:
            validation_data = pd.read_csv(path)
        else:
            validation_data = pd.read_excel(path, sheet_name=self.sheet)
        validation_data = validation_data[['Latitude', 'Longitude', 'Radius']]
        validation_data = validation_data[(validation_data["Latitude"] >= data_range[0]) &  # filter for testing zone
                                        (validation_data["Latitude"] <= data_range[1]) &
                                        (validation_data["Longitude"] >= data_range[2]) &
                                        (validation_data["Longitude"] <= data_range[3])]
        validation_data  = validation_data[["Radius", "Latitude", "Longitude"]]
        return validation_data

    def matchPoints(self) -> pd.DataFrame:
        """
        adds values to indicate if the point is a true seamount
        Parameters
        ----------
        out_data : pd.DataFrame
            Data to add values to
        Returns
        -------
        pd.DataFrame
            Data with values added. True seamounts are marked with 1
            while points that are not seamounts are marked with -1
        """
        if self.training_data is None:
            raise AttributeError("Training data has not been added to the class yet")
        return pd.DataFrame(self.training_data,  # type: ignore
                            columns=["Latitude", "Longitude", "Radius", "TrueSeamount"])

    @abstractmethod
    def scoreTestData(self, path) -> float:
        """
        Scores the data on a testing zone outside of the training zone
        Parameters
        ----------
        data_range : tuple
            Range of data to score in the form (min_lat, max_lat, min_lon, max_lon)
        args : any
            Any additional arguments needed to score the data
        Returns
        -------
        float
            Score of the test data
        """

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
        return _SeamountSupport.RADIUS * c

    @staticmethod
    def _pythagorean(lat1, lon1, lat2, lon2) -> float:
        """
        Calculates the pythagorean distance between two points
        on given coordinates in meters
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
        x = lon2 - lon1
        y = lat2 - lat1
        return np.sqrt(x ** 2 + y ** 2) / 1000

    @staticmethod
    def formatData(data, zval) -> np.ndarray:
        """
        Formats the data to be used in an algorithm
        by converting to UTM coordinates and filtering
        out the unneeded columns
        Parameters
        ----------
        data : pd.DataFrame
            Data to format
        Returns
        -------
        np.ndarray
            Formatted data
        """
        data = data[['Latitude', 'Longitude', zval]]
        data = data[["Latitude", "Longitude", zval]]
        data["TrueSeamount"] = np.zeros(data.shape[0])
        return data.to_numpy()
