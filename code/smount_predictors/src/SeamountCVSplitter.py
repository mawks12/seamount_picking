"""
Splitting module to split the seamounts into training and testing datasets
"""

from math import sqrt, log2
from typing import Generator, Any
import numpy as np
from sklearn.model_selection import BaseCrossValidator
import pandas as pd


class SeamountCVSplitter(BaseCrossValidator):
    """
    Splits the seamounts into training and a testing datasets.
    Leaves some overlap betweet splits to ensure that features
    are not split between the two sets.
    """

    def __init__(self, n_splits: int = 4, pad: int = 5) -> None:
        """
        Initialize the SeamountCVSplitter object.

        Args:
            n_splits (int): The number of splits to generate. for splitting purposes, must be a square number.
            pad (int): The padding value to add to each split.

        Raises:
            ValueError: If n_splits is an odd number.

        Returns:
            None
        """
        super().__init__()
        if sqrt(n_splits) % 1 != 0 or sqrt(n_splits) % 2 != 0:
            raise ValueError("n_splits must be a square number with an even root.")
        self.n_splits = n_splits
        self.pad = pad

    def split(self, X, y, groups=None):
        """
        Splits the data into training and testing datasets.

        Parameters:
            X (array-like): The input data array.
            y (array-like): Target variable.
            groups: Optional group variable (not used in this method).

        Returns:
            train_index, test_index: The training and testing indices.
        """
        yield [None, None]  # TODO: Implement this method

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splits.

        Returns:
            n_splits: The number of splits.
        """
        return self.n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """
        Returns the testing indices for each split.

        Parameters:
            X (array-like): The input data array.
            y (array-like): Target variable.
            groups: Optional group variable (not used in this method).

        Returns:
            test_index: The testing indices.
        """
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        for _, test_index in self.split(X, y, groups):
            yield test_index
