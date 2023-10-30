"""LSTSQ model."""

import numpy as np
from numpy.linalg import lstsq

from ._functions import _add_intercept
from ._linear_model import _LinearModel


class _LSTSQ(_LinearModel):
    """Linear regression (least squares) model."""

    def __init__(self, fit_intercept: bool = False) -> None:
        """Initialize LSTSQ model.

        :param fit_intercept: True if fit intercept in model, defaults to False
        :type fit_intercept: bool, optional
        """
        super().__init__(fit_intercept)

    def fit(self, X: np.array, y: np.array):
        """Fit model.

        :param X: input matrix
        :type X: np.array
        :param y: dependent variable
        :type y: np.array
        :return: fitted model
        :rtype: _LSTSQ
        """
        if self.fit_intercept:
            X = _add_intercept(X)
        lstsq_beta, _, _, _ = lstsq(X, y, rcond=-1)
        self._assign_coef_and_intercept(lstsq_beta)
        return self
