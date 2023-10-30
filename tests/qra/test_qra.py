"""Test cases for the QRA model."""
from typing import Tuple

import numpy as np
import pytest
from sklearn.linear_model import QuantileRegressor

from remodels.qra import QRA

from . import sample_data


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
