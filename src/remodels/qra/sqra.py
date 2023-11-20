"""SQRA model."""

import numpy as np

from ._functions import _sqra
from .qra import QRA


class SQRA(QRA):
    """SQRA."""

    def __init__(
        self, quantile: float = 0.5, H: float = None, fit_intercept: bool = False
    ) -> None:
        """Initialize SQRA model.

        :param quantile: quantile
        :type quantile: float
        :param fit_intercept: True if fit intercept in model, defaults to False
        :type fit_intercept: bool, optional
        :param H: smoothing parameter called the bandwidth, must be positive
            real number; if None, it is automatically estimated using Scott's
            (or Silverman's) rule-of-thumb
        :type H: float
        """
        super().__init__(quantile=quantile, fit_intercept=fit_intercept)
        self.H = H

    def fit(self, X: np.array, y: np.array):
        """Fit model.

        :param X: input matrix
        :type X: np.array
        :param y: dependent variable
        :type y: np.array
        :return: fitted model
        :rtype: SQRA
        """
        beta = _sqra(X, y, self.quantile, self.H, self.fit_intercept)
        self._assign_coef_and_intercept(beta)
        return self
