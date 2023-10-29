"""Test cases for the sFQRA model."""
from typing import Tuple

import numpy as np
import pytest

from remodels.qra import FQRA
from remodels.qra import sFQRA

from . import sample_data


def test_sfqra_predicitons(sample_data: Tuple[np.array, np.array]) -> None:
    """sFQRA model predictions equals FQRA predictions with prepared data."""
    X_, y_ = sample_data
    mean = np.mean(X_, axis=1)
    std = np.std(X_, axis=1)

    X = (X_ - mean[:, np.newaxis]) / std[:, np.newaxis]
    y = (y_ - mean) / std
    y_fqra = FQRA(0.5, 1, True).fit(X, y).predict(X)
    y_fqra = y_fqra * std + mean

    y_sfqra = sFQRA(0.5, 1, True).fit(X_, y_).predict(X_)

    assert y_sfqra == pytest.approx(y_fqra)
