"""FQRM model."""

import numpy as np

from ._lstsq import _LSTSQ
from .fqra import FQRA
from .qra import QRA


class FQRM(FQRA):
    """FQRM."""

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
