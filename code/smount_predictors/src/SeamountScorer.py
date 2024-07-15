"""
Custom module for seamount scoring.
"""

from typing import Callable
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.neighbors import BallTree
import numpy as np


class SeamountScorer:
    """
    Custom class for scoring seamount prediction problems.
    """

    # TODO: pick a default tolerance value
    def __init__(self, y_centers, tolerance: float = 15, scoring: str | Callable[..., float] = 'recall') -> None:
        """
        Initializes a SeamountScorer object.

        Args:
            y_centers (np.ndarray): The true values, as an array of lat, lon pairs representing
            the center of each seamount.
            tolerance (float): The tolerance value for the scorer.
            scoring (str | Callable[..., float]): The scoring method for the scorer. Can be a string or a callable.
            Default is 'log'. If a callable is passed, the scorer will use a custom scoring method. If a string is
            passed, the scorer will use the corresponding scoring method, which can be one of 'log', 'mse', 'mae', or
            'rmse'.

        Attributes:
            tolerance (float): The tolerance value for the scorer.
            scoring (str): The scoring method for the scorer.
            score (Callable[..., float]): The scoring method for the scorer.

        Returns:
            None
        """
        scoring_methods = {'log', 'mse', 'mae', 'rmse', 'recall'}
        self.tolerance = tolerance
        self.y_centers = y_centers
        if isinstance(scoring, Callable):
            self.score = scoring
            self.scoring = 'custom'
            return
        if scoring not in scoring_methods:
            raise ValueError(f"Scoring method must be one of {scoring_methods}")
        self.scoring = scoring
        self.score = self._get_scorer()

    def _get_scorer(self) -> Callable[..., float]:
        """
        Returns the scoring method based on the scoring attribute.

        Returns:
            scoring_method: The scoring method based on the scoring attribute.
        """
        scorers = {
            'log': self._log_loss,
            'mse': self._mse,
            'mae': self._mae,
            'rmse': self._rmse,
            'recall': self._recall
        }
        return scorers[self.scoring]

    def _map_nearest(self, y_true, y_pred) -> np.ndarray:
        """
        Generates predicted values that can be used to calculate the 
        classifier recall.

        Args:
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            np.ndarray: The adjusted predicted values.
        """
        y_true = np.radians(y_true)
        y_pred = np.radians(y_pred)
        tree = BallTree(y_pred, metric='haversine')
        num_near = tree.query_radius(y_true, r=1.30725161e-5, count_only=True)
        y_pred = np.where(num_near >= self.tolerance, 1, 0)  # type: ignore

        assert len(y_pred) == len(y_true), "Length mismatch"
        return y_pred

    def __call__(self, estimator, X: np.ndarray, y_true: np.ndarray) -> float:
        if len(y_true) == 0:
            raise RuntimeError("Split containted not valid samples")
        y_pred = X[estimator.predict(X) == 1][:, :2]
        if len(y_pred) == 0:
            return -1 * float('inf')
        y_pred = self._map_nearest(self.y_centers, y_pred)
        if len(y_pred) == 0:
            return -1 * float('inf')
        return self.score(np.repeat(1, len(y_pred)), y_pred)

    def _log_loss(self, y_true, y_pred) -> float:
        return float(log_loss(y_true, y_pred))

    def _mse(self, y_true, y_pred) -> float:
        return float(mean_squared_error(y_true, y_pred))

    def _mae(self, y_true, y_pred) -> float:
        return float(mean_absolute_error(y_true, y_pred))

    def _rmse(self, y_true, y_pred) -> float:
        return np.sqrt(self._mse(y_true, y_pred))

    def _recall(self, y_true, y_pred) -> float:
        return float(np.sum(y_true == y_pred) / len(y_true))
