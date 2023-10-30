"""Test cases for the sFQRM model."""
from typing import Tuple

import numpy as np
import pytest

from remodels.qra import FQRM
from remodels.qra import sFQRM

from . import sample_data


def test_sfqrm_predicitons(sample_data: Tuple[np.array, np.array]) -> None:
    """sFQRM model predictions equals FQRM predictions with prepared data."""
    X_, y_ = sample_data
    mean = np.mean(X_, axis=1)
    std = np.std(X_, axis=1)

    X = (X_ - mean[:, np.newaxis]) / std[:, np.newaxis]
    y = (y_ - mean) / std
    y_fqrm = FQRM(0.5, 1, True).fit(X, y).predict(X)
    y_fqrm = y_fqrm * std + mean

    y_sfqrm = sFQRM(0.5, 1, True).fit(X_, y_).predict(X_)

    assert y_sfqrm == pytest.approx(y_fqrm)
