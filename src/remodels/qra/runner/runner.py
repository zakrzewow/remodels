"""*QR* runner."""

import concurrent.futures
from itertools import repeat

import numpy as np

from remodels.qra import QRA


def _process(X_train, y_train, X_test, q, qra_model):
    qra_model.quantile = q / 100
    y_test = qra_model.fit(X_train, y_train).predict(X_test)
    return q, y_test


_default_qra_model = QRA(fit_intercept=True)


class Runner:
    """Runner."""

    def __init__(
        self,
        calibration_window: int = 14,
        prediction_window: int = 1,
        qra_model=_default_qra_model,
        max_workers: int = None,
    ) -> None:
        """Init runner.

        :param calibration_window: length of calibration window, defaults to 14
        :type calibration_window: int, optional
        :param prediction_window: length of prediction window, defaults to 1
        :type prediction_window: int, optional
        :param qra_model: *QR* model, defaults to QRA(fit_intercept=True)
        :type qra_model: QRA, optional
        :param max_workers: process pool executor max workers, defaults to None
        :type max_workers: int, optional
        """
        self.calibration_window = calibration_window
        self.prediction_window = prediction_window
        self.qra_model = qra_model
        self.max_workers = max_workers

    def fit_predict(self, X: np.array, y: np.array) -> "_Results":
        """Fit predict.

        :param X: data matrix
        :type X: np.array
        :param y: endogenous variable
        :type y: np.array
        :return: Results object
        :rtype: _Results
        """
        executor = concurrent.futures.ProcessPoolExecutor(self.max_workers)

        self.prediction_window = 10
        Y_pred = np.zeros((X.shape[0] - self.calibration_window, 99), np.float_)

        for i in range(
            0,
            X.shape[0] - self.calibration_window - self.prediction_window + 1,
            self.prediction_window,
        ):
            X_train = X[i : i + self.calibration_window]
            y_train = y[i : i + self.calibration_window]
            X_test = X[
                i
                + self.calibration_window : i
                + self.calibration_window
                + self.prediction_window
            ]

            for q, y_test in executor.map(
                _process,
                repeat(X_train),
                repeat(y_train),
                repeat(X_test),
                range(1, 100),
                repeat(self.qra_model),
            ):
                Y_pred[i : i + self.prediction_window, q - 1] = y_test

        return _Results(
            Y_pred,
            y[self.calibration_window :],
            self.prediction_window,
        )


class _Results:
    """Results."""

    def __init__(
        self,
        Y_pred: np.array,
        y_test: np.array,
        prediction_window: int,
    ) -> None:
        """Results.

        :param Y_pred: matrix of predictions
        :type Y_pred: np.array
        :param y_test: endogenous variable
        :type y_test: np.array
        :param prediction_window: length of prediction window
        :type prediction_window: int
        """
        self.Y_pred = Y_pred
        self.y_test = y_test
        self.prediction_window = prediction_window

    def aec(self, alpha: int) -> float:
        """Average empirical coverage.

        :param alpha: length of prediction interval
        :type alpha: int
        :return: average empirical coverage value
        :rtype: float
        """
        hits = self._hits(alpha)
        return hits.sum() / self.Y_pred.shape[0]

    def ec_h(self, alpha: int) -> np.array:
        """Empirical coverage per 'hour'.

        :param alpha: length of prediction interval
        :type alpha: int
        :return: emipirical coverage per hour values
        :rtype: np.array
        """
        hits = self._hits(alpha)
        d = self.Y_pred.shape[0] // self.prediction_window
        ec_h = np.zeros(shape=(d,))
        for i in range(d):
            ec_h[i] = hits[i :: self.prediction_window].sum() / d
        return ec_h

    def ec_mad(self, alpha: int) -> float:
        """Empirical coverage per 'hour' MAD.

        :param alpha: length of prediction interval
        :type alpha: int
        :return: mean absolute deviation of empirical coverage per hour values
        :rtype: float
        """
        return np.mean(np.abs(self.ec_h(alpha) - alpha / 100))

    def _hits(self, alpha) -> np.array:
        lower_bound = 49 - alpha // 2
        upper_bound = 49 + alpha // 2
        return (self.Y_pred[:, lower_bound] <= self.y_test) & (
            self.y_test <= self.Y_pred[:, upper_bound]
        )

    def aps(self) -> float:
        """Aggregate pinball score.

        :return: aggregate pinball score value
        :rtype: float
        """
        return np.mean(self._pinball_score_matrix())

    def aps_extreme_quantiles(self, n_quantiles: int) -> float:
        """Aggregate pinball score for n extreme quantiles.

        :param n_quantiles: number of leftmost and rightmost quantiles
        :type n_quantiles: int
        :return: aggregate pinball score computed for extreme quantiles
        :rtype: float
        """
        pinball_score_matrix = self._pinball_score_matrix()
        idx = list(range(n_quantiles)) + list(range(99 - n_quantiles, 99))
        return np.mean(pinball_score_matrix[:, idx])

    def _pinball_score_matrix(self) -> np.array:
        q = np.arange(0.01, 1.00, 0.01)
        q = np.expand_dims(q, axis=0)
        q = np.repeat(q, self.Y_pred.shape[0], axis=0)

        resid = self.y_test[:, np.newaxis] - self.Y_pred
        return (q - (resid < 0)) * resid
