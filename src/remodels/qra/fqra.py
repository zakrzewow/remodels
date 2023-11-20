"""FQRA model."""

from typing import Tuple

import numpy as np

from ._linear_model import _LinearModel
from .qra import QRA


class FQRA(QRA):
    """FQRA."""

    def __init__(
        self, quantile: float = 0.5, n_factors: int = None, fit_intercept: bool = False
    ) -> None:
        """Initialize FQRA model.

        :param quantile: quantile
        :type quantile: float
        :param n_factors: number of factors (principal components) used
        :type n_factors: int
        :param fit_intercept: True if fit intercept in model, defaults to False
        :type fit_intercept: bool, optional
        """
        super().__init__(quantile, fit_intercept)
        self.n_factors = n_factors

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
