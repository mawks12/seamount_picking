"""
Transformer module for using gausian convolution and gradient
peaks to identify seamounts
"""

import numpy as np
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter, sobel
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class SeamountTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer class for using gausian convolution and gradient peaks
    to identify seamounts. Convolves a gaussian filter with the data
    and applies a sobel filter to find the gradient peaks. The transformed
    data is the product of the gradient and the smoothed data, and can
    be passed to a linear kernel SVM for classification.
    """
    CONV_KERN = np.array([
        [0.125, 0.125, 0.125],
        [0.125, 0.000, 0.125],
        [0.125, 0.125, 0.125]
        ])

    def __init__(self, sigma=0.6) -> None:
        """
        Initializes a SeamountTransformer object.

        Returns:
            None
        """
        self.sigma = sigma
        self.scalar = StandardScaler()

    def fit(self, X=None, y=None) -> 'SeamountTransformer':
        """
        Fits the transformer to the input data array.

        Returns:
            self: The SeamountTransformer object.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input data array by applying a gaussian filter and a sobel filter,
        and multiplying the gradient with the smoothed data.

        Returns:
            transformed: The transformed data array.
        """
        df = pd.DataFrame(X, columns=['lat', 'lon', 'z'])
        df.set_index(['lat', 'lon'], inplace=True)
        data_index = df.index
        if df.index.duplicated().any():
            df = df[~df.index.duplicated()]
        Xxr = df.to_xarray()
        Xxr['z'] = Xxr['z'].fillna(0)
        # Scaling for y axis. NOTE: since this uses the mean, the larger the area
        # the less accurate the scaling. Works best in small areas.
        x1_sig = self.sigma / np.cos(np.radians(Xxr['lat'].values.mean()))
        # Filtering to smooth data, and find gradient peaks
        smoothed = gaussian_filter(Xxr['z'].values, sigma=(self.sigma, x1_sig))
        y_grad = sobel(smoothed, axis=0)
        x_grad = sobel(smoothed, axis=1)
        grad = np.hypot(x_grad, y_grad)
        transed = smoothed * grad
        # Multiplying the gradient with the smoothed data to leave only high gradient
        # gaussian peaks, such as seamounts
        numpy_array = convolve(transed, self.CONV_KERN, mode='same')
        # Previous step leaves rings, so convolve to reconstruct
        # the center of those rings. NOTE: This does grow the area
        Xxr['z'].values = numpy_array
        df = Xxr.to_dataframe()
        good_vals = df.loc[data_index].reset_index()
        numpy_array = good_vals.values
        if np.isnan(numpy_array).any():
            print(numpy_array[np.isnan(numpy_array)])
            raise ValueError("NaN values in the transformed data")
        if hasattr(self.scalar, 'mean_'):
            numpy_array[:, 2] = self.scalar.transform(numpy_array[:, 2].reshape(-1, 1)).flatten()  # type: ignore
        else:
            numpy_array[:, 2] = self.scalar.fit_transform(numpy_array[:, 2].reshape(-1, 1)).flatten()
        assert not (numpy_array[:, 2] == 0).all(), "All values are zero"
        return numpy_array[:, 2].reshape(-1, 1)  # type: ignore
