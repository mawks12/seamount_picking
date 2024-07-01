"""
Splitting class to split the seamounts into training and testing datasets
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

    def _split_helper(self, data: pd.DataFrame, y: np.ndarray, split_depth: int, split_num: int) -> pd.DataFrame:
        """
        Recursively splits the data into portions for cross validation.

        Args:
            data (pd.DataFrame): The input data array.
            y (np.ndarray): Target variable.
            split_depth (int): The depth of the split.
        """
        if split_depth == 0:  # base case - if depth is 0, label and return the split
            data['split'] = split_num
            return data
        center = y.mean(axis=0)  # get the center of mass for the labels
        q3 = self._split_helper(  # recursively split the data into quadrants along the center
            data[(data['lat'] < center + self.pad) & (data['lon'] < center + self.pad + self.pad)],
            # Modify the center of the split such that the each split has padding according to the pad value
            y[(y[:, 0] < center + self.pad) & (y[:, 1] < center + self.pad)], split_depth - 1, split_num + 1
            )
        q4 = self._split_helper(
            data[(data['lat'] < center + self.pad) & (data['lon'] >= center - self.pad)],
            y[(y[:, 0] < center + self.pad) & (y[:, 1] >= center - self.pad)], split_depth - 1, split_num + 2
            )
        q2 = self._split_helper(
            data[(data['lat'] >= center - self.pad) & (data['lon'] < center + self.pad)],
            y[(y[:, 0] >= center - self.pad) & (y[:, 1] < center + self.pad)], split_depth - 1, split_num + 3
            )
        q1 = self._split_helper(
            data[(data['lat'] >= center - self.pad) & (data['lon'] >= center - self.pad)],
            y[(y[:, 0] >= center - self.pad) & (y[:, 1] >= center - self.pad)], split_depth - 1, split_num + 4
            )
        return pd.concat([q1, q2, q3, q4])

    def split(self, X: np.ndarray, y: np.ndarray, groups=None) -> Generator[tuple[Any, Any], Any, None]:
        """
        Splits the data into training and testing datasets.

        Parameters:
            X (array-like): The input data array.
            y (array-like): Target variable.
            groups: Optional group variable (not used in this method).

        Returns:
            train_index, test_index: The training and testing indices.
        """
        sorting = pd.DataFrame(X, columns=['lon', 'lat', 'z'])
        sorting['index'] = np.arange(len(sorting))
        sorting['split'] = np.nan
        sorting = self._split_helper(sorting, y, int(log2(self.n_splits) // 2), 0)
        for i in sorting['split'].unique():
            train_index = sorting[sorting['split'] != i]['index'].values
            test_index = sorting[sorting['split'] == i]['index'].values
            yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splits.

        Returns:
            n_splits: The number of splits.
        """
        return self.n_splits

    def _iter_test_indices(self, X, y, groups=None) -> Generator[Any, Any, None]:
        """
        Returns the testing indices for each split.

        Parameters:
            X (array-like): The input data array.
            y (array-like): Target variable.
            groups: Optional group variable (not used in this method).

        Returns:
            test_index: The testing indices.
        """
        for _, test_index in self.split(X, y, groups):
            yield test_index
