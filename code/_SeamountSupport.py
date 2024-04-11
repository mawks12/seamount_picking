"""
Basic support class for the seamount detection code
"""

from abc import abstractmethod
import math
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
    FILTERTHRSHMIN = -1000.5  # threshold for filtering out points
    FILTERTHRDHMAX = 2008.2
    BOUNDARY = 2  # Distance from boundary to be considered a boundary point
    # Used in scoring so that boundary points are not penalized or rewarded

    def __init__(self, validation_data: str, train_zone=(-90, 90, -180, 180), sheet: str="new mask") -> None:
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
        self.seamount_points = None
        seamounts = self.filterData(validation_data, self.train_zone)
        self.num_seamounts = seamounts.shape[0]
        self.distance = _SeamountSupport._pythagorean  # distance function
        self.datascaler = StandardScaler()
        self.unlabled_data = None
        self.label_hash = {}
        self.training_data = np.array([])
        self.p_neighbors = scipy.spatial.KDTree([[0]])
        self.seamount_dict = {}
        self.__points = np.array([])

    def _trueSeamount(self, test_points) -> int:
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
            1 if true seamount, 0 if boundary, -1 if not a seamount
        """
        if self.unlabled_data is None or self.label_hash is None:
            raise AttributeError("No training data has been added")
        hashed = self.label_hash.get(tuple(test_points), -2)
        if hashed == -2:
            raise KeyError(f"{test_points} not found in training data")
        return hashed

    def addTrainingData(self, training_data: np.ndarray) -> None:
        """
        Adds training data to the class
        Parameters
        ----------
        training_data : np.ndarray of the form (lat, lon, zval)
            Data to add to the class
        Returns
        -------
        None
        """
        training_data = training_data[training_data[:, 0] >= self.train_zone[0]]  # filter for training zone
        training_data = training_data[training_data[:, 0] <= self.train_zone[1]]
        training_data = training_data[training_data[:, 1] >= self.train_zone[2]]
        training_data = training_data[training_data[:, 1] <= self.train_zone[3]]
        self.training_data = training_data
        self.datascaler.fit(training_data)
        seamounts = self.filterData(self.validation_path, self.train_zone, csv=False)
        self.__points = seamounts.to_numpy()  # get points
        self.seamount_dict = dict(zip(zip(self.__points[:, 0], self.__points[:, 1]), \
                                      self.__points[:, 2]))
        # dictionary of true seamounts and radii for faster distance checking
        self.p_neighbors = scipy.spatial.KDTree(self.__points[:, :2])
        for i in range(self.training_data.shape[0]):
            self.training_data[i][3] = self._radiusMatch(
                training_data[i], self.p_neighbors, self.__points, self.seamount_dict)
        self.seamount_points = self.training_data[self.training_data[:, 3] == 1].shape[0]
        self.training_data = self.datascaler.transform(training_data)
        self.training_data = self.training_data[  # type: ignore
            self.training_data[:, 2] > _SeamountSupport.FILTERTHRSHMIN]  # type: ignore
        self.training_data = self.training_data[self.training_data[:, 2] < _SeamountSupport.FILTERTHRDHMAX]
        #  filter out points that have too low a gravity value
        self.unlabled_data = self.training_data[:, :3]  # type: ignore
        assert isinstance(self.unlabled_data, np.ndarray)
        self.label_hash = dict(zip(map(  # create hashtable for faster checking
            tuple, self.unlabled_data[:, :2]), training_data[:, 3]))

    def getNear(self, point: tuple[float, float]):
        """
        Gets the radius of the nearest seamount
        Parameters
        ----------
        point : tuple
            Point to get the radius for
        Returns
        -------
        radius : int
            Radius of the nearest seamount
        nearest : tuple
            Nearest seamount
        """
        _, i = self.p_neighbors.query([point])
        nearest = self.__points[i][0][:2]
        radius = self.seamount_dict.get(tuple(nearest), -1)
        assert radius != -1
        return tuple(nearest), radius

    def _radiusMatch(self, test_points, tree, points, query) -> int:
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
            1 if true seamount, 0 if boundary, -1 if not a seamount
        """
        _, i = tree.query([test_points[0], test_points[1]])
        nearest = points[i]  # get nearest point
        radius = query.get((nearest[0], nearest[1]), -10)
        if radius == -10:  # check if point is in the dictionary - all points should be
            # if not raise an error
            raise KeyError(f"Error: {nearest[0]}, {nearest[1]} not found in seamounts")
        dist = _SeamountSupport._haversine(nearest[0], nearest[1], test_points[0], test_points[1])
        if dist < (radius):
            return 1
        return -1

    def filterData(self, path, data_range: tuple, csv=False) -> pd.DataFrame:
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
        validation_data  = validation_data[["Latitude", "Longitude", "Radius"]]
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

    def getPindex(self, ind: int) -> tuple[float, float]:
        """
        Get the index of a point in the seamount.

        Parameters
        ---------
        ind: int
            The index of the point.
        Returns
        -------
        A tuple containing the x and y coordinates of the point.
        """
        return tuple(self.__points[ind])  # type: ignore
    
    def getValidationSeamounts(self) -> np.ndarray:
        """
        Get the seamounts in the validation data
        Returns
        -------
        np.ndarray
            Seamounts in the validation data
        """
        ...  # TODO: Implement this method

    @abstractmethod
    def scoreTestData(self, test_data) -> float:
        """
        Scores the data on a testing zone outside of the training zone
        Parameters
        ----------
        test_data : array-like
            Data to score
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
        c = 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a))
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
    def formatData(data: pd.DataFrame, zval: str) -> np.ndarray:
        """
        Formats the data to be used in an algorithm
        by adding a column of zeros to the end of the data
        for label assignment
        Parameters
        ----------
        data : pd.DataFrame
            Data to format
        zval : str
            name of the last column of data
        Returns
        -------
        np.ndarray
            Formatted data
        """
        data = data[["Latitude", "Longitude", zval]]
        data.loc[:, "TrueSeamount"] = np.zeros(data.shape[0])
        return data.to_numpy()

    @staticmethod
    def seamountNorm(radius: int, dist: float) -> float:
        """
        Distributiuon function for p_dist calculation
        Parameters
        ----------
        radius : float
            Radius of the seamount
        point : tuple
            Point to calculate the probability for
        Returns
        -------
        float
            Probability that the point is part of the seamount
        """
        assert dist is not None
        try:
            return math.exp((2.447 * dist) ** 2 / (2 * (radius ** 2)))
        except OverflowError:
            return 0

    @staticmethod
    def pDist(radius: int, center: tuple[float, float], point: tuple[float, float]) -> float:
        """
        Calculates the probability that a point is part of a seamount given
        the center of the seamount and the radius. Uses an adapted normal distribution
        such that the probability is 1 at the center of the seamount and 0.05 at the edge
        for any given radius
        Parameters
        ----------
        radius : float
            Radius of the seamount
        center : tuple
            Center of the seamount in the form (lat, lon)
        point : tuple
            Point to calculate the probability for
        Returns
        -------
        float
            Probability that the point is part of the seamount
        """
        if radius == -1:
            raise ValueError("Point is not a seamount")
        dist = _SeamountSupport._haversine(center[0], center[1], point[0], point[1])
        return _SeamountSupport.seamountNorm(radius, dist)
