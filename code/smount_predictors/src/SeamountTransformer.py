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
    TEST_CONV_KERN = np.array([
        [0.04166667, 0.04166667, 0.04166667, 0.04166667, 0.04166667],
        [0.04166667, 0.04166667, 0.04166667, 0.04166667, 0.04166667],
        [0.04166667, 0.04166667, 0.00000000, 0.04166667, 0.04166667],
        [0.04166667, 0.04166667, 0.04166667, 0.04166667, 0.04166667],
        [0.04166667, 0.04166667, 0.04166667, 0.04166667, 0.04166667]
        ])

    def __init__(self, sigma=1) -> None:
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
        # FIXME: return to using xarrays, and simply fill nans with 0 to avoid errors
        df = pd.DataFrame(X, columns=['lat', 'lon', 'z'])
        df.set_index(['lat', 'lon'], inplace=True)
        data_index = df.index
        if df.index.duplicated().any():
            df = df[~df.index.duplicated()]  # TODO: Find how duplicates are being created
            # They seem to all be true duplicates, so this should be fine, but it's worth investigating
        Xxr = df.to_xarray()
        Xxr['z'] = Xxr['z'].fillna(0)
        x1_sig = self.sigma / np.cos(Xxr['lat'].values.mean())
        smoothed = gaussian_filter(Xxr['z'].values, sigma=(self.sigma, x1_sig))
        y_grad = sobel(smoothed, axis=0)
        x_grad = sobel(smoothed, axis=1)
        grad = np.hypot(x_grad, y_grad)
        transed = smoothed * grad
        numpy_array = convolve(transed, self.TEST_CONV_KERN, mode='same')
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
        return numpy_array[:, 2].reshape(-1, 1)  # type: ignore
