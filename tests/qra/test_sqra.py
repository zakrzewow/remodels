"""Test cases for the SQRA model."""
from typing import Tuple

import numpy as np
import pytest

from remodels.qra import SQRA

from . import sample_data


def test_sqra_predicitons(sample_data: Tuple[np.array, np.array]) -> None:
    """Sum of SQRA model predictions equals specific number."""
    X_, y_ = sample_data
    assert SQRA(0.5, 1, False).fit(X_, y_).predict(X_).sum() == pytest.approx(
        989.015207045325, abs=1e-2
    )
    assert SQRA(0.5, 1, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        990.8340987938492, abs=1e-2
    )
    assert SQRA(0.5, None, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        989.6052326758037, abs=1e-2
    )
