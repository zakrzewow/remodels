"""*QR* tester."""

import concurrent.futures
from itertools import repeat
from typing import Callable

import numpy as np
from scipy.stats import chi2

from remodels.qra import QRA


def _process(X_train, y_train, X_test, q, qra_model):
    qra_model.quantile = q / 100
    y_test = qra_model.fit(X_train, y_train).predict(X_test)
    return q, y_test


_default_qr_model = QRA(fit_intercept=True)


class QR_Tester:
    """QR Tester."""

    def __init__(
        self,
        calibration_window: int = 14,
        prediction_window: int = 1,
        qr_model=_default_qr_model,
        max_workers: int = None,
    ) -> None:
        """Init tester.

        :param calibration_window: length of calibration window, defaults to 14
        :type calibration_window: int, optional
        :param prediction_window: length of prediction window, defaults to 1
        :type prediction_window: int, optional
        :param qr_model: *QR* model, defaults to QRA(fit_intercept=True)
        :type qr_model: QRA, optional
        :param max_workers: process pool executor max workers, defaults to None
        :type max_workers: int, optional
        """
        self.calibration_window = calibration_window
        self.prediction_window = prediction_window
        self.qr_model = qr_model
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
                repeat(self.qr_model),
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
        ec_h = np.zeros(shape=(self.prediction_window,))
        for i in range(self.prediction_window):
            hits_h = hits[i :: self.prediction_window]
            ec_h[i] = hits_h.sum() / hits_h.shape[0]
        return ec_h

    def ec_mad(self, alpha: int) -> float:
        """Empirical coverage per 'hour' MAD.

        :param alpha: length of prediction interval
        :type alpha: int
        :return: mean absolute deviation of empirical coverage per hour values
        :rtype: float
        """
        return np.mean(np.abs(self.ec_h(alpha) - alpha / 100))

    def kupiec_test(self, alpha: int, significance_level: float = 0.05) -> int:
        """Kupiec test.

        :param alpha: length of predition interval
        :type alpha: int
        :param significance_level: test significance level, defaults to 0.05
        :type significance_level: float, optional
        :return: number of hours that test is not rejected
        :rtype: int
        """
        return self._count_hypothesis_not_rejected(
            self._kupiec_test_statistic, alpha, significance_level, 1
        )

    def christoffersen_test(self, alpha: int, significance_level: float = 0.05) -> int:
        """Christoffersen test.

        :param alpha: length of predition interval
        :type alpha: int
        :param significance_level: test significance level, defaults to 0.05
        :type significance_level: float, optional
        :return: number of hours that test is not rejected
        :rtype: int
        """
        return self._count_hypothesis_not_rejected(
            self._christoffersen_test_statistic, alpha, significance_level, 2
        )

    def _count_hypothesis_not_rejected(
        self,
        test_func: Callable[[np.array, float], float],
        alpha: int,
        significance_level: float,
        df: int,
    ) -> int:
        alpha_p = alpha / 100
        hits = self._hits(alpha)
        hypothesis_not_rejected_counter = 0

        for i in range(self.prediction_window):
            hits_h = hits[i :: self.prediction_window]
            test_statistic = test_func(hits_h, alpha_p)
            is_not_rejected = test_statistic < chi2.ppf(1 - significance_level, df=df)
            hypothesis_not_rejected_counter += is_not_rejected

        return hypothesis_not_rejected_counter

    def _kupiec_test_statistic(self, hits: np.array, alpha_p: float) -> float:
        n = hits.shape[0]
        n1 = hits.sum()
        n0 = n - n1
        L_0 = n1 * np.log(alpha_p) + n0 * np.log(1 - alpha_p)
        L_A = n1 * np.log(n1 / n) + n0 * np.log(n0 / n)
        return 2 * (L_A - L_0)

    def _christoffersen_test_statistic(self, hits: np.array, alpha_p: float) -> float:
        t00 = (~hits & ~self.__shift_arr(hits, -1)).sum()
        t01 = (~hits & self.__shift_arr(hits, -1)).sum()
        t10 = (hits & ~self.__shift_arr(hits, -1)).sum()
        t11 = (hits & self.__shift_arr(hits, -1)).sum()

        p01 = t01 / (t00 + t01)
        p11 = t11 / (t11 + t10)
        L_A = (
            t00 * np.log(1 - p01)
            + t01 * np.log(p01)
            + t10 * np.log(1 - p11)
            + t11 * np.log(p11)
        )
        L_0 = (t00 + t10) * np.log(1 - alpha_p) + (t01 + t11) * np.log(alpha_p)
        return 2 * (L_A - L_0)

    @staticmethod
    def __shift_arr(arr, num, fill_value=np.nan):
        result = np.empty_like(arr)
        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result[:] = arr
        return result

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
