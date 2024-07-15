"""
Transformer module for using gausian convolution and gradient
peaks to identify seamounts
"""

import numpy as np
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter, sobel
from sklearn.preprocessing import StandardScaler
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
        self.sig = 1.0
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
        smoothed = gaussian_filter(X, sigma=self.sig)
        y_grad = sobel(smoothed, axis=0)
        x_grad = sobel(smoothed, axis=1)
        grad = np.hypot(x_grad, y_grad)
        numpy_array = smoothed * grad
        numpy_array = convolve(numpy_array, self.TEST_CONV_KERN, mode='same')
        if np.isnan(numpy_array).any():
            print(numpy_array[np.isnan(numpy_array)])
            raise ValueError("NaN values in the transformed data")
        if hasattr(self.scalar, 'mean_'):
            numpy_array[:, 2] = self.scalar.transform(numpy_array[:, 2].reshape(-1, 1)).flatten()
        else:
            numpy_array[:, 2] = self.scalar.fit_transform(numpy_array[:, 2].reshape(-1, 1)).flatten()
        return numpy_array[:, 2].reshape(-1, 1)  # type: ignore
