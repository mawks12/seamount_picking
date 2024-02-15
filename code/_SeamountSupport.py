"""
Basic support class for the seamount detection code
"""

from abc import abstractmethod
from math import atan2
import scipy.spatial as sps
import pandas as pd
import numpy as np

class _SeamountSupport:
    """
    Basic Support class for the seamount detection code
    Contains basic training, testing, and distance functions, as well
    as all of the general static methods used in the seamount
    detection code
    """
    MARGIN = 0.002  # percentage margin allowed to be considered a seamount cluster
    RADIUS = 6371  # radius of the earth in km

    def __init__(self, validation_data, fast=False, train_zone=(-90, 90, -180, 180), sheet: str="new mask") -> None:
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
        self.seamounts = pd.read_excel(validation_data, sheet_name=sheet)
        self.seamounts = self.seamounts[['Latitude', 'Longitude', 'Radius']]
        self.seamounts = self.seamounts[(self.seamounts["Latitude"] >= train_zone[0]) &  # filter for trianing zone
                                        (self.seamounts["Latitude"] <= train_zone[1]) &
                                        (self.seamounts["Longitude"] >= train_zone[2]) &
                                        (self.seamounts["Longitude"] <= train_zone[3])]
        self.num_seamounts = self.seamounts.shape[0]
        self.__points = self.seamounts.to_numpy()  # get points
        self.seamount_dict = dict(zip(zip(self.__points[:, 0], self.__points[:, 1]), \
                                      self.__points[:, 2]))
        # dictionary of true seamounts and radii for faster distance checking
        self.p_neighbors = sps.KDTree(self.__points[:, :2])
        self.global_points_set = set(map(tuple, self.__points))  # set of true seamounts
        self.distance = _SeamountSupport._haversine if not fast else _SeamountSupport._pythagorean  # distance function

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

    def _filterTestData(self, data_range: tuple) -> pd.DataFrame:
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
        validion_data = pd.read_excel(self.validation_path, sheet_name=self.sheet)
        validion_data = validion_data.drop(columns=["VGG Height", "base_depth", "-",
                                            "Name", "Charted", "surface_depth"])
        validion_data = validion_data[(validion_data["Latitude"] >= data_range[0]) &  # filter for testing zone
                                        (validion_data["Latitude"] <= data_range[1]) &
                                        (validion_data["Longitude"] >= data_range[2]) &
                                        (validion_data["Longitude"] <= data_range[3])]
        return validion_data

    @abstractmethod
    def scoreTestData(self, data_range: tuple, path, params, test_data,  *args) -> float:
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
        return np.sqrt(x ** 2 + y ** 2) * _SeamountSupport.RADIUS
