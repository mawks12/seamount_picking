"""
Transform Merging class
"""

import warnings
import numpy as np
from SeamountTransformer import SeamountTransformer


class TransformMerger(SeamountTransformer):
    """
    Version of SeamountTransformer designed to process
    multiple non continous blocks of data

    TODO: find a way for this to work with pipe
    Also maybe fix original to work with overloads?
    """

    def transform(self, X: list[np.ndarray]) -> np.ndarray:
        transformed = np.array([])
        for data in X:
            if data.shape[0] == 0:
                warnings.warn(
                    'An empty array was passed to the transformer'
                )
                continue
            processed = super().transform(data)
            np.append(transformed, processed)
        return processed
