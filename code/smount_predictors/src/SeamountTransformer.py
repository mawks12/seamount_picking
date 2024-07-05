"""
Transformer module for using gausian convolution and gradient
peaks to identify seamounts
"""

from typing import Self
import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter, sobel
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin


class SeamountTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer class for using gausian convolution and gradient peaks
    to identify seamounts. Convolves a gaussian filter with the data
    and applies a sobel filter to find the gradient peaks. The transformed
    data is the product of the gradient and the smoothed data, and can
    be passed to a linear kernel SVM for classification.
    """
    TEST_CONV_KERN = np.array([
        [0.02370913, 0.06444810, 0.06438915, 0.02370913, 0.06444810],
        [0.06444810, 0.02370913, 0.02370913, 0.06444810, 0.02370913],
        [0.02370913, 0.02370913, 0.02370913, 0.02370913, 0.02370913],
        [0.02370913, 0.06444810, 0.06438915, 0.02370913, 0.06444810],
        [0.06444810, 0.02370913, 0.02370913, 0.06444810, 0.02370913]
        ])

    def __init__(self) -> None:
        """
        Initializes a SeamountTransformer object.

        Returns:
            None
        """
        self.xar = None
        self.sig = 1.0

    def fit(self, X: np.ndarray) -> Self:
        """
        Fits the SeamountTransformer transformer to the given data.

        Parameters:
            X (xr.DataArray): The input data array containing the lon, lat, and z variables.

        Returns:
            self: The fitted SeamountTransformer transformer.
        """
        lon = X[:, 0]
        lat = X[:, 1]
        zvalue = X[:, 2]
        df = pd.DataFrame({
            'lon': lon,
            'lat': lat,
            'zvalue': zvalue
        })
        ds = xr.Dataset.from_dataframe(df.set_index(['lon', 'lat']))
        self.xar = ds['zvalue']
        return self

    def transform(self, _=None) -> np.ndarray:
        """
        Transforms the input data array by applying a gaussian filter and a sobel filter,
        and multiplying the gradient with the smoothed data.

        Returns:
            transformed: The transformed data array.
        """
        if not isinstance(self.xar, xr.DataArray):
            raise AttributeError("Transformer has not been fitted yet")
        smoothed = gaussian_filter(self.xar.data, sigma=self.sig)
        y_grad = sobel(smoothed, axis=0)
        x_grad = sobel(smoothed, axis=1)
        grad = np.hypot(x_grad, y_grad)
        self.xar.data = grad * smoothed
        self.xar.data = convolve(self.xar.data, self.TEST_CONV_KERN, mode='same')
        ds_reset = self.xar.to_dataset()
        df = ds_reset.to_dataframe().reset_index()
        numpy_array = df[['lon', 'lat', 'zvalue']].to_numpy()
        return numpy_array
