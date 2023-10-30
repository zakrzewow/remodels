"""Test cases for the LQRA model."""
from typing import Tuple

import numpy as np
import pytest
from sklearn.linear_model import QuantileRegressor

from remodels.qra import LQRA

from . import sample_data


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
