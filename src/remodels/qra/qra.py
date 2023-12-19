"""QRA model."""

import numpy as np

from ._functions import _lqra
from ._linear_model import _LinearModel


class QRA(_LinearModel):
    r"""A class that represents the QRA model.

    The QRA model is a simple quantile regression model.
    Fitting a quantile regression model involves solving a minimaztion problem:

    .. math::
        \hat{\beta_k} = \underset{\beta \in \mathbb{R}^n}{\operatorname{argmin}} \left\{ \sum_{i=1}^{t} \rho_k (Y_i - X_i \beta) \right\}

    where :math:`\rho_k` is given by:

    .. math::
        \rho_k (e) = e (k - {1}_{(e < 0)} )

    and :math:`k` is the fixed quantile.
    """

    def __init__(self, quantile: float = 0.5, fit_intercept: bool = False) -> None:
        """Initialize the QRA model.

        :param quantile: quantile
        :type quantile: float
        :param fit_intercept: True if fit intercept in model, defaults to False
        :type fit_intercept: bool, optional
        """
        self.quantile = quantile
        super().__init__(fit_intercept)

    def fit(self, X: np.array, y: np.array):
        """Fit the model to the data.

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
