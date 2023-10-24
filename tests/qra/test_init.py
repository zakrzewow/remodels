"""Test cases for the qra.__init__ module."""
from typing import Tuple

import numpy as np
import pytest

from remodels.qra import QRA
from remodels.qra import QRM


@pytest.fixture
def sample_data() -> Tuple[np.array, np.array]:
    """Fixture providing sample data to test QRA models."""
    np.random.seed(0)
    X_ = np.random.uniform(0, 10, size=(100, 2))
    y_ = X_[:, 0] + X_[:, 1] + np.random.normal(0, 1, size=(100,))
    return X_, y_


def test_qra_predicitons(sample_data: Tuple[np.array, np.array]) -> None:
    """Sum of QRA model predictions equals specific number."""
    X_, y_ = sample_data
    assert QRA(0.5, False).fit(X_, y_).predict(X_).sum() == pytest.approx(
        982.0416688210856
    )
    assert QRA(0.5, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        984.5833310199318
    )


def test_qrm_predicitons(sample_data: Tuple[np.array, np.array]) -> None:
    """Sum of QRM model predictions equals specific number."""
    X_, y_ = sample_data
    assert QRM(0.5, False).fit(X_, y_).predict(X_).sum() == pytest.approx(
        976.5882131381337
    )
    assert QRM(0.5, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        979.7208019501684
    )
