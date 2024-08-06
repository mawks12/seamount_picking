"""
A wrapper for svms allowing the scoring methods to use all data dimensions, but only the z dimension for prediction.
"""
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin

class SeamountSVM(BaseEstimator,  TransformerMixin):
    """
    Custom class for scoring seamount prediction problems.
    """
    def __init__(self, **kwargs) -> None:
        """
        Initializes a SeamountSVM object.

        Args:
            C (float): Regularization parameter. The strength of the regularization is inversely proportional to C. \
                Must be strictly positive. The penalty is a squared l2 penalty.
            kernel (str): Specifies the kernel type to be used in the algorithm. \
                It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable. If none is given, \
                'rbf' will be used.
            degree (int): Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
            gamma (str): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma='scale' (default) is passed \
                then it uses 1 / (n_features * X.var()) as value of gamma, if 'auto', uses 1 / n_features.
            coef0 (float): Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
            shrinking (bool): Whether to use the shrinking heuristic.
            probability (bool): Whether to enable probability estimates. This must be enabled prior to calling fit, \
                and will slow down that method.
            tol (float): Tolerance for stopping criterion.
            cache_size (float): Specify the size of the kernel cache (in MB).
            class_weight (dict | str | None): Set the parameter C of class i to class_weight[i]*C for SVC. \
                If not given, all classes are supposed to have weight one. The 'balanced' mode uses the values \
                of y to automatically adjust weights inversely proportional to class frequencies
            verbose (bool): Enable verbose output. Note that this setting takes advantage of a per-process \
                runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.

        Returns:
            None
        """
        self.svm = SVC(**kwargs)

    def fit(self, X, y):
        """
        Fit the SVM model.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target values.

        Returns:
            self: The fitted SeamountSVM object.
        """
        trans_X = X[:, 2]
        self.svm.fit(trans_X, y)
        return self

    def predict(self, X):
        """
        Predict the target values.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted target values.
        """
        return self.svm.predict(X[:, 2])
    