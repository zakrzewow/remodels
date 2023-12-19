"""SQRA model."""

import numpy as np

from ._functions import _sqra
from .qra import QRA


class SQRA(QRA):
    r"""A class that represents the SQRA model.

    The SQRA model is a QRA model with a loss function smoothed by a kernel density estimator:

    .. math::
        \hat{\beta_k} = \underset{\beta \in \mathbb{R}^n}{\operatorname{argmin}} \left\{ \sum_{i=1}^{t} \left( H \cdot \phi \left( \frac{Y_i - X_i \beta}{H} \right) + \left( k - \Phi \left( - \frac{Y_i - X_i \beta}{H}\right) \right) \left( Y_i - X_i \beta \right) \right) \right\}

    where :math:`H` is a bandwidth parameter.
    """

    def __init__(
        self, quantile: float = 0.5, H: float = None, fit_intercept: bool = False
    ) -> None:
        """Initialize the SQRA model.

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
        """Fit the model to the data.

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
