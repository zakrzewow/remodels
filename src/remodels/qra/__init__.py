"""QRA methods."""

from typing import Tuple

import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import linprog
from scipy.optimize import minimize
from scipy.stats import iqr
from scipy.stats import norm


def _add_intercept(X: np.array) -> np.array:
    new_X = np.ones(shape=(X.shape[0], X.shape[1] + 1))
    new_X[:, 1:] = X
    return new_X


def _lqra(X, y, quantile: float, lambda_: float = 0.0, fit_intercept: bool = False):
    # input matrix X - N rows x K columns
    N, K = X.shape

    if fit_intercept:
        K += 1
        X = _add_intercept(X)

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

    optimize_result = linprog(
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
    # for now, TNC (non-linear) minimizer is used

    if fit_intercept:
        X = _add_intercept(X)

    qra = QRA(quantile=quantile, fit_intercept=False).fit(X, y)
    beta__initial_guess = qra._beta
    if H is None:  # rule-of-thumb estimation
        residuals = y - qra.predict(X)
        resid_std = np.std(residuals)
        resid_iqr = iqr(residuals)
        H = min(resid_std, resid_iqr / 1.38898) * 1.06 * (X.shape[0] ** (-1 / 5))

    def rho(beta):
        residuals = y - X @ beta
        return (
            np.sum(H * norm.pdf(residuals / H))
            + (quantile - norm.cdf(-residuals / H)) @ residuals
        )

    beta__initial_guess = _lqra(X, y, quantile=quantile, lambda_=0, fit_intercept=False)
    return minimize(rho, beta__initial_guess, method="TNC").x


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


class QRA(_LinearModel):
    """QRA."""

    def __init__(self, quantile: float, fit_intercept: bool = False) -> None:
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


class QRM(QRA):
    """QRM."""

    def __init__(self, quantile: float, fit_intercept: bool = False) -> None:
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


class FQRA:
    """FQRA."""

    def __init__(
        self, quantile: float, n_factors: int, fit_intercept: bool = False
    ) -> None:
        """Initialize FQRA model.

        :param quantile: quantile
        :type quantile: float
        :param n_factors: number of factors (principal components) used
        :type n_factors: int
        :param fit_intercept: True if fit intercept in model, defaults to False
        :type fit_intercept: bool, optional
        """
        self.quantile = quantile
        self.n_factors = n_factors
        self.fit_intercept = fit_intercept

    def fit(self, X: np.array, y: np.array):
        """Fit model.

        :param X: input matrix
        :type X: np.array
        :param y: dependent variable
        :type y: np.array
        :return: fitted model
        :rtype: FQRA
        """
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X: np.array) -> np.array:
        """Predict dependent variable.

        :param X: input matrix
        :type X: np.array
        :return: prediction
        :rtype: np.array
        """
        X_train_f, X_test_f = self._get_factors(X)

        if self.n_factors is None:
            self.n_factors = self._select_best_n_factors_with_bic(
                X_train_f,
                QRA(self.quantile, self.fit_intercept),
            )

        X_train_f = X_train_f[:, : self.n_factors]
        X_test_f = X_test_f[:, : self.n_factors]
        qra = QRA(self.quantile, self.fit_intercept).fit(X_train_f, self._y_train)
        return qra.predict(X_test_f)

    def _get_factors(self, X: np.array) -> Tuple[np.array, np.array]:
        X_full = np.concatenate([self._X_train, X], axis=0)
        _, _, Vh = np.linalg.svd(X_full, full_matrices=False)
        F = (X_full @ Vh.T) / X.shape[1]
        return F[: self._X_train.shape[0], :], F[self._X_train.shape[0] :, :]

    def _select_best_n_factors_with_bic(
        self,
        X_train_f: np.array,
        model: _LinearModel,
    ) -> int:
        last_bic = np.inf
        best_n_factors = 0
        N, K = X_train_f.shape

        for i in range(1, K + 1):
            model = model.fit(X_train_f[:, :i], self._y_train)
            y_hat_train = model.predict(X_train_f[:, :i])
            rss = np.sum(np.square(self._y_train - y_hat_train))
            bic = N * np.log(rss / N) + i * np.log(N)
            if bic > last_bic:
                break
            last_bic = bic
            best_n_factors = i
        return best_n_factors


class FQRM(FQRA):
    """FQRM."""

    def __init__(
        self, quantile: float, n_factors: int, fit_intercept: bool = False
    ) -> None:
        """Initialize FQRA model.

        :param quantile: quantile
        :type quantile: float
        :param n_factors: number of factors (principal components) used
        :type n_factors: int
        :param fit_intercept: True if fit intercept in model, defaults to False
        :type fit_intercept: bool, optional
        """
        super().__init__(quantile, n_factors, fit_intercept)

    def fit(self, X: np.array, y: np.array):
        """Fit model.

        :param X: input matrix
        :type X: np.array
        :param y: dependent variable
        :type y: np.array
        :return: fitted model
        :rtype: FQRM
        """
        return super().fit(X, y)

    def predict(self, X: np.array) -> np.array:
        """Predict dependent variable.

        :param X: input matrix
        :type X: np.array
        :return: prediction
        :rtype: np.array
        """
        X_train_f, X_test_f = self._get_factors(X)

        if self.n_factors is None:
            self.n_factors = self._select_best_n_factors_with_bic(
                X_train_f,
                _LSTSQ(self.fit_intercept),
            )

        X_train_f = X_train_f[:, : self.n_factors]
        X_test_f = X_test_f[:, : self.n_factors]

        lstsq = _LSTSQ(self.fit_intercept).fit(X_train_f, self._y_train)
        X_train_fm = lstsq.predict(X_train_f)
        X_train_fm = np.expand_dims(X_train_fm, axis=1)
        X_test_fm = lstsq.predict(X_test_f)
        X_test_fm = np.expand_dims(X_test_fm, axis=1)

        qra = QRA(self.quantile, self.fit_intercept).fit(X_train_fm, self._y_train)
        return qra.predict(X_test_fm)


class LQRA(QRA):
    """LQRA."""

    def __init__(
        self,
        quantile: float,
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


class SQRA(QRA):
    """SQRA."""

    def __init__(
        self,
        quantile: float,
        H: float = None,
        fit_intercept: bool = False,
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


class SQRM(SQRA):
    """SQRM."""

    def __init__(
        self,
        quantile: float,
        H: float = 0.0,
        fit_intercept: bool = False,
    ) -> None:
        """Initialize SQRM model.

        :param quantile: quantile
        :type quantile: float
        :param fit_intercept: True if fit intercept in model, defaults to False
        :type fit_intercept: bool, optional
        :param H: smoothing parameter called the bandwidth
        :type H: float
        """
        super().__init__(quantile=quantile, H=H, fit_intercept=fit_intercept)

    def fit(self, X: np.array, y: np.array):
        """Fit model.

        :param X: input matrix
        :type X: np.array
        :param y: dependent variable
        :type y: np.array
        :return: fitted model
        :rtype: SQRM
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
