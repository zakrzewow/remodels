"""LQRA model."""

import numpy as np

from ._functions import _lqra
from .qra import QRA


class LQRA(QRA):
    r"""A class that represents the LQRA model.

    The LQRA model is a quantile regression model with a linear penalty factor added to the loss function:

    .. math::
        \hat{\beta_k} = \underset{\beta \in \mathbb{R}^n}{\operatorname{argmin}} \left\{ \sum_{i=1}^{t} \rho_k (Y_i - X_i \beta) + \lambda \sum_{i=1}^{n} |\beta_i| \right\}

    where :math:`\lambda` is a regularization parameter.
    """

    def __init__(
        self,
        quantile: float = 0.5,
        lambda_: float = 0.0,
        fit_intercept: bool = False,
    ) -> None:
        """Initialize the LQRA model.

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
        """Fit the model to the data.

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
