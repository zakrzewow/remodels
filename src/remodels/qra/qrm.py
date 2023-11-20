"""QRM model."""

import numpy as np

from .qra import QRA


class QRM(QRA):
    """QRM."""

    def __init__(self, quantile: float = 0.5, fit_intercept: bool = False) -> None:
        """Initialize QRM model.

        :param quantile: quantile
        :type quantile: float
        :param fit_intercept: True if fit intercept in model, defaults to False
        :type fit_intercept: bool, optional
        """
        super().__init__(quantile=quantile, fit_intercept=fit_intercept)

    def fit(self, X: np.array, y: np.array):
        """Fit model.

        :param X: input matrix
        :type X: np.array
        :param y: dependent variable
        :type y: np.array
        :return: fitted model
        :rtype: QRM
        """
        X = np.mean(X, axis=1, keepdims=True)
        return super().fit(X, y)

    def predict(self, X: np.array) -> np.array:
        """Predict dependent variable.

        :param X: input matrix
        :type X: np.array
        :return: prediction
        :rtype: np.array
        """
        X = np.mean(X, axis=1, keepdims=True)
        return super().predict(X)
