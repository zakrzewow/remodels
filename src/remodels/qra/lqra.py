"""LQRA model."""

import numpy as np

from ._functions import _lqra
from .qra import QRA


class LQRA(QRA):
    """LQRA."""

    def __init__(
        self,
        quantile: float = 0.5,
        lambda_: float = 0.0,
        fit_intercept: bool = False,
    ) -> None:
        """Initialize LQRA model.

        :param quantile: quantile
        :type quantile: float
        :param fit_intercept: True if fit intercept in model, defaults to False
        :type fit_intercept: bool, optional
        :param lambda_: LASSO regularization parameter
        :type lambda_: float
        """
        super().__init__(quantile=quantile, fit_intercept=fit_intercept)
        self.lambda_ = lambda_

    def fit(self, X: np.array, y: np.array):
        """Fit model.

        :param X: input matrix
        :type X: np.array
        :param y: dependent variable
        :type y: np.array
        :return: fitted model
        :rtype: LQRA
        """
        beta = _lqra(X, y, self.quantile, self.lambda_, self.fit_intercept)
        self._assign_coef_and_intercept(beta)
        return self
