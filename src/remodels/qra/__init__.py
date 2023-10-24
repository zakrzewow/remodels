"""QRA methods."""

import numpy as np
from scipy.optimize import linprog as scipy_linprog
from scipy.optimize import minimize
from scipy.stats import norm


def _lqra(X, y, quantile: float, lambda_: float = 0.0, fit_intercept: bool = False):
    # input matrix X - N rows x K columns
    N, K = X.shape

    if fit_intercept:
        K += 1
        X = np.concatenate([np.ones((N, 1)), X], axis=1)

    # scaling regularization parameter to make it sample-independent
    lambda_ = N * lambda_

    c = np.concatenate(
        [
            np.full((2 * K,), fill_value=lambda_),
            np.full((N,), fill_value=quantile),
            np.full((N,), fill_value=1 - quantile),
        ]
    )
    if fit_intercept:  # do not penalize intercept
        c[0] = 0
        c[K] = 0

    b_eq = y
    A_eq = np.concatenate([X, -X, np.eye(N), -np.eye(N)], axis=1)

    optimize_result = scipy_linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        method="highs",  # highs is an optimalized solver (simplex or interior-point)
    )

    # beta = beta+ - beta-
    beta = optimize_result.x[:K] - optimize_result.x[K : 2 * K]
    return beta


def _sqra(X, y, quantile: float, H: float, fit_intercept=False):
    # SQRA can be solved with gradient-descent method (2022-he)

    if fit_intercept:
        N = X.shape[0]
        X = np.concatenate([np.ones((N, 1)), X], axis=1)

    def rho(beta):
        residuals = y - X @ beta
        return (
            np.sum(H * norm.pdf(residuals / H))
            + (quantile - norm.cdf(-residuals / H)) @ residuals
        )

    beta__initial_guess = _lqra(X, y, quantile=quantile, lambda_=0, fit_intercept=False)
    return minimize(rho, beta__initial_guess, method="TNC").x


class QRA:
    """QRA."""

    def __init__(self, quantile: float, fit_intercept: bool = False) -> None:
        """Initializes QRA model.

        :param quantile: quantile
        :type quantile: float
        :param fit_intercept: True if fit intercept in QRA model, defaults to False
        :type fit_intercept: bool, optional
        """
        self.quantile = quantile
        self.fit_intercept = fit_intercept

    def fit(self, X: np.array, y: np.array) -> "QRA":
        """Fits QRA model.

        :param X: input matrix
        :type X: np.array
        :param y: dependent variable
        :type y: np.array
        :return: fitted QRA model
        :rtype: QRA
        """
        beta = _lqra(X, y, self.quantile, 0, self.fit_intercept)
        self._assign_coef_and_intercept(beta)
        return self

    def _assign_coef_and_intercept(self, beta: np.array):
        if self.fit_intercept:
            self._coef = beta[1:]
            self._intercept = beta[0]
        else:
            self._coef = beta
            self._intercept = 0

    def predict(self, X: np.array) -> np.array:
        """Predicts dependent variable.

        :param X: input matrix
        :type X: np.array
        :return: prediction
        :rtype: np.array
        """
        return X @ self._coef + self._intercept


class QRM(QRA):
    """QRM."""

    def __init__(self, quantile: float, fit_intercept: bool = False) -> None:
        """Initializes QRM model.

        :param quantile: quantile
        :type quantile: float
        :param fit_intercept: True if fit intercept in QRA model, defaults to False
        :type fit_intercept: bool, optional
        """
        super().__init__(quantile=quantile, fit_intercept=fit_intercept)

    def fit(self, X: np.array, y: np.array) -> "QRM":
        """Fits QRM model.

        :param X: input matrix
        :type X: np.array
        :param y: dependent variable
        :type y: np.array
        :return: fitted QRM model
        :rtype: QRM
        """
        X = np.mean(X, axis=1, keepdims=True)
        return super().fit(X, y)

    def predict(self, X: np.array) -> np.array:
        """Predicts dependent variable.

        :param X: input matrix
        :type X: np.array
        :return: prediction
        :rtype: np.array
        """
        X = np.mean(X, axis=1, keepdims=True)
        return super().predict(X)
