"""Linear model."""

import numpy as np
import pandas as pd


class _LinearModel:
    """Common abstract class for all linear models."""

    def __init__(self, fit_intercept: bool = False) -> None:
        """Initialize linear model.

        :param fit_intercept: True if fit intercept in model, defaults to False
        :type fit_intercept: bool, optional
        """
        self.fit_intercept = fit_intercept

    def fit(self, X: np.array, y: np.array) -> "_LinearModel":
        """Fit model.

        :param X: input matrix
        :type X: np.array
        :param y: dependent variable
        :type y: np.array
        :return: fitted model
        :rtype: _LinearModel
        """
        return self

    def predict(self, X: np.array) -> np.array:
        """Predict dependent variable.

        :param X: input matrix
        :type X: np.array
        :return: prediction
        :rtype: np.array
        """
        return X @ self._coef + self._intercept

    def _assign_coef_and_intercept(self, beta: np.array):
        self._beta = beta
        if self.fit_intercept:
            self._coef = beta[1:]
            self._intercept = beta[0]
        else:
            self._coef = beta
            self._intercept = 0
