"""Test cases for the QRM model."""
from typing import Tuple

import numpy as np
import pytest
from sklearn.linear_model import QuantileRegressor

from remodels.qra import QRM

from . import sample_data


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
