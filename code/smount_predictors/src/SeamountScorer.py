"""
Custom module for seamount scoring.
"""

from typing import Callable
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.neighbors import BallTree
import numpy as np
import xarray as xr


class SeamountScorer:
    """
    Custom class for scoring seamount prediction problems.
    TODO: identify best scoring method for seamount prediction problems.
    """

    # TODO: pick a default tolerance value
    def __init__(self, tolerance: float = 0.1, scoring: str | Callable[..., float] = 'log') -> None:
        """
        Initializes a SeamountScorer object.

        Args:
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
        scoring_methods = ['log', 'mse', 'mae', 'rmse']
        self.tolerance = tolerance
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
            'rmse': self._rmse
        }
        return scorers[self.scoring]

    def _map_nearest(self, y_true, y_pred) -> np.ndarray:
        """
        Maps the nearest true value to each predicted value, as long
        as the distance between the predicted value and the true value
        is less than the tolerance value.

        Args:
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            np.ndarray: The adjusted predicted values.
        """
        y_true = np.radians(y_true)
        tree = BallTree(np.array([y_true]), metric='haversine')
        dist, _ = tree.query(np.array([y_pred]))
        y_pred = np.array([y_true[i] if d < self.tolerance else y_pred[i] for i, d in enumerate(dist)])
        return y_pred

    def __call__(self, estimator, X: xr.DataArray, y_true: np.ndarray) -> float:
        y_pred = estimator.predict(X)
        y_pred = self._map_nearest(y_true, y_pred)
        return self.score(y_true, y_pred)

    def _log_loss(self, y_true, y_pred) -> float:
        return float(log_loss(y_true, y_pred))

    def _mse(self, y_true, y_pred) -> float:
        return float(mean_squared_error(y_true, y_pred))

    def _mae(self, y_true, y_pred) -> float:
        return float(mean_absolute_error(y_true, y_pred))

    def _rmse(self, y_true, y_pred) -> float:
        return np.sqrt(self._mse(y_true, y_pred))
