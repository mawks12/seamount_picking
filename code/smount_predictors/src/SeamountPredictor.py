"""
Predictor class for using gausian convolution and gradient
peaks to identify seamounts
"""

from typing import Self
import numpy as np
from scipy.ndimage import gaussian_filter, label, sobel, center_of_mass
import xarray as xr
from sklearn.base import BaseEstimator, ClassifierMixin


class SeamountPredictor(BaseEstimator, ClassifierMixin):
    """
    Predictor class for using gausian convolution and gradient peaks
    to identify seamounts. Convolves a gaussian filter with the data
    and applies a sobel filter to find the gradient peaks. The
    peaks are then classified as seamounts based on the
    product of the gradient and the gaussian filter.
    This estimator requires the use of xarray DataArrays,
    and is meant to be used in conjunction with the
    SeamountSplitter and SeamountScorer classes when used
    with sklearn pipelines.
    """

    def __init__(self, classifier_threshold) -> None:
        """
        Initializes a SeamountPredictor object.

        Args:
            classifier_threshold (float): The threshold value for the classifier.

        Attributes:
            classifier_threshold (float): The threshold value for the classifier.
            lon (None): The longitude of the seamount.
            lat (None): The latitude of the seamount.
            elevation (None): The elevation of the seamount.
            locations (None): The locations of the seamount.

        Returns:
            None
        """
        self.classifier_threshold = classifier_threshold
        self.lon = None
        self.lat = None
        self.elevation = None
        self.locations = None

    def fit(self, X: xr.DataArray, y=None) -> Self:
        """
        Fits the SeamountPredictor model to the given data.

        Parameters:
            X (xr.DataArray): The input data array containing the lon, lat, and z variables.
            y: Optional target variable (not used in this method).

        Returns:
            self: The fitted SeamountPredictor model.
        """
        self.lon = X['lon'].values
        self.lat = X['lat'].values
        self.elevation = X['z'].values
        return self

    def transform(self, X: xr.DataArray) -> Self:
        """
        Transforms the input data array by applying a series of operations.

        Args:
            X (xr.DataArray): The input data array.

        Returns:
            SeamountPredictor: The transformed SeamountPredictor object.
        """
        if self.lon is None or self.lat is None or self.elevation is None:
            raise AttributeError("Model has not been fitted yet")
        smoothed = gaussian_filter(self.elevation, sigma=1.2)  # FIXME: find optimal sigma value
        y_grad = sobel(smoothed, axis=0)
        x_grad = sobel(smoothed, axis=1)
        grad = np.hypot(x_grad, y_grad)
        transformed = grad * smoothed
        transformed = transformed > self.classifier_threshold
        labeled_array, num_features = label(transformed)  # type: ignore
        seamounts = center_of_mass(transformed, labeled_array, range(1, num_features + 1))
        self.locations = np.array([[self.lat[round(lon)], self.lon[round(lat)]] for lat, lon in seamounts])
        return self

    def predict(self, X: xr.DataArray):# -> NDArray[Any] | None:
        """
        Predicts the seamounts in the given data array.

        Parameters:
            X (xr.DataArray): The input data array.

        Returns:
            np.ndarray: The predicted seamounts. Note that the
            coordinates are in the form of (lat, lon) for compatibility
            with the SeamountScorer and haversine_distances functions.
        """
        self.transform(X)
        return self.locations
