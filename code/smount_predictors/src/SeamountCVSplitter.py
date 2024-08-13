"""
Splitting module to split the seamounts into training and testing datasets
"""

import warnings
from sklearn.neighbors import BallTree
from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd


class SeamountCVSplitter(BaseCrossValidator):
    """
    Splits the seamounts into training and a testing datasets,
    using a stratified approach to ensure the same proportion
    of seamounts in each split.
    """

    def __init__(self, n_splits: int = 4) -> None:
        """
        Initialize the SeamountCVSplitter object.

        Args:
            n_splits (int): The number of splits to generate.
        """
        super().__init__()
        self.n_splits = n_splits
        self.split_size = 0
        self.true_split_size = 0
        self.false_split_size = 0
        self.data = pd.DataFrame()
        self.splits_generated = 0

    def split(self, X, y, groups=None):
        """
        Makes a splits of the data by selecting a random index and finding
        the len(X) // n_splits closest points to that index, drawing
        only from the points that have not been chosen yet. The split
        is stratified to get the same proportion of seamount labled pixels
        in each split.

        Parameters:
            X (array-like): The input data array.
            y (array-like): Target variable.
            groups: Optional group variable (not used in this method).

        Returns:
            train_index, test_index: The training and testing indices.
        """
        X = np.array(X)
        y = np.array(y)
        for test_index in self._iter_test_indices(X, y, groups):
            train_index = np.setdiff1d(np.arange(len(X)), test_index)
            yield train_index, test_index

    def _make_split(self):
        """
        Makes a single split of the data

        Returns:
            test_index: The testing indices.
        """
        if self.data['chosen'].all():  # check if all splits have been generated
            assert self.splits_generated == self.n_splits, "Split count does not match chosen"
            raise RuntimeError('All train-test splits have been generated')
        if self.splits_generated == self.n_splits - 1:
            assert self.data['chosen'].any(), "Split count mismatch with chosen"
            last_split = self.data[~self.data['chosen']].index
            self.data.loc[last_split, 'chosen'] = True  # modify the last split to avoid over-generating splits
            self.splits_generated += 1
            return last_split
        true_tree = BallTree(
            self.data[(~self.data['chosen']) & (self.data['true'] == 1)]
            [['lon', 'lat']].values,
            leaf_size=2, metric='haversine'
            )
        false_tree = BallTree(
            self.data[(~self.data['chosen']) & (self.data['true'] == 0)]
            [['lon', 'lat']].values,
            leaf_size=2,
            metric='haversine'
            )
        true_seed = np.random.choice(self.data[(~self.data['chosen']) & (self.data['true'] == 1)].index)
        false_seed = false_tree.query(self.data[['lon', 'lat']].loc[true_seed].to_numpy().reshape(1, -1), k=1)[1][0][0]
        # Generates two seed points to grow splits from from, such that each split will have the
        # same proportion of seamounts and non-seamounts as the original data. picks false point
        # based on smallest distance to true point, to ensure that the split is not unevenly
        # spacialy distributed
        _, true_indices = true_tree.query(
            self.data[['lon', 'lat']]
            .loc[true_seed]
            .to_numpy().reshape(1, -1),
            k=self.true_split_size
            )
        _, false_indices = false_tree.query(
            self.data[['lon', 'lat']].loc[false_seed]
            .to_numpy().reshape(1, -1),
            k=self.false_split_size
            )
        indecies = np.concatenate([true_indices[0], false_indices[0]])
        # merges the two sets of indices to get the final testing indices with the correct proportion of seamounts
        indices = self.data[~self.data['chosen']].iloc[indecies].index
        if self.data.loc[indices, 'chosen'].any():
            raise RuntimeError("Index already chosen")
        self.data.loc[indices, 'chosen'] = True
        self.splits_generated += 1
        return indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splits.

        Returns:
            n_splits: The number of splits.
        """
        return self.n_splits

    def _iter_test_indices(self, X, y, groups=None):
        """
        Returns the testing indices for each split.

        Parameters:
            X (array-like): The input data array.
            y (array-like): Target variable.
            groups: Optional group variable (not used in this method).

        Returns:
            test_index: The testing indices.
        """
        X = np.array(X)
        y = np.array(y)
        self.split_size = len(X) // self.n_splits
        self.true_split_size = int((sum(y) / len(y)) * self.split_size)
        self.false_split_size = self.split_size - self.true_split_size
        self.data = pd.DataFrame(X[:, :2], columns=['lat', 'lon'])
        self.data['chosen'] = False
        self.data['true'] = y
        for _ in range(self.n_splits):
            yield self._make_split()
