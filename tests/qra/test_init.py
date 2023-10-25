"""Test cases for the qra.__init__ module."""
from typing import Tuple

import numpy as np
import pytest
from sklearn.linear_model import QuantileRegressor

from remodels.qra import LQRA
from remodels.qra import QRA
from remodels.qra import QRM
from remodels.qra import SQRA
from remodels.qra import SQRM


@pytest.fixture
def sample_data() -> Tuple[np.array, np.array]:
    """Fixture providing sample data to test QRA models."""
    np.random.seed(0)
    X_ = np.random.uniform(0, 10, size=(100, 2))
    y_ = X_[:, 0] + X_[:, 1] + np.random.normal(0, 1, size=(100,))
    return X_, y_


def test_qra_predicitons(sample_data: Tuple[np.array, np.array]) -> None:
    """QRA model predictions are equal to sklearn QR implementation predictions."""
    X_, y_ = sample_data

    qra_predictions = QRA(0.5, True).fit(X_, y_).predict(X_)
    sklearn_qr_predictions = (
        QuantileRegressor(quantile=0.5, alpha=0, fit_intercept=True, solver="highs")
        .fit(X_, y_)
        .predict(X_)
    )
    assert qra_predictions == pytest.approx(sklearn_qr_predictions)


def test_qrm_predicitons(sample_data: Tuple[np.array, np.array]) -> None:
    """QRM model predictions are equal to sklearn QR implementation predictions."""
    X_, y_ = sample_data

    qrm_predictions = QRM(0.5, True).fit(X_, y_).predict(X_)

    X_mean = np.mean(X_, axis=1, keepdims=True)
    sklearn_qr_predictions = (
        QuantileRegressor(quantile=0.5, alpha=0, fit_intercept=True, solver="highs")
        .fit(X_mean, y_)
        .predict(X_mean)
    )
    assert qrm_predictions == pytest.approx(sklearn_qr_predictions)


def test_lqra_predicitons(sample_data: Tuple[np.array, np.array]) -> None:
    """LQRA model predictions are equal to sklearn QR implementation predictions."""
    X_, y_ = sample_data

    lqra_predictions = LQRA(0.5, 1, True).fit(X_, y_).predict(X_)
    sklearn_qr_predictions = (
        QuantileRegressor(quantile=0.5, alpha=1, fit_intercept=True, solver="highs")
        .fit(X_, y_)
        .predict(X_)
    )
    assert lqra_predictions == pytest.approx(sklearn_qr_predictions)


def test_sqra_predicitons(sample_data: Tuple[np.array, np.array]) -> None:
    """Sum of SQRA model predictions equals specific number."""
    X_, y_ = sample_data
    assert SQRA(0.5, 1, False).fit(X_, y_).predict(X_).sum() == pytest.approx(
        989.015207045325
    )
    assert SQRA(0.5, 1, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        990.8340987938492
    )
    assert SQRA(0.5, None, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        989.6052326758037
    )


def test_sqrm_predicitons(sample_data: Tuple[np.array, np.array]) -> None:
    """Sum of SQRM model predictions equals specific number."""
    X_, y_ = sample_data
    assert SQRM(0.5, 1, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        991.161454057788
    )
    assert SQRM(0.5, None, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        990.1044896031061
    )
