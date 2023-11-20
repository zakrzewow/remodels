"""QRA model."""

import numpy as np

from ._functions import _lqra
from ._linear_model import _LinearModel


class QRA(_LinearModel):
    """QRA."""

    def __init__(self, quantile: float = 0.5, fit_intercept: bool = False) -> None:
        """Initialize QRA model.

        :param quantile: quantile
        :type quantile: float
        :param fit_intercept: True if fit intercept in model, defaults to False
        :type fit_intercept: bool, optional
        """
        self.quantile = quantile
        super().__init__(fit_intercept)

    def fit(self, X: np.array, y: np.array):
        """Fit model.

        :param X: input matrix
        :type X: np.array
        :param y: dependent variable
        :type y: np.array
        :return: fitted model
        :rtype: QRA
        """
        beta = _lqra(X, y, self.quantile, 0, self.fit_intercept)
        self._assign_coef_and_intercept(beta)
        return self
