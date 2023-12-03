"""Test cases for the SQRM model."""
from typing import Tuple

import numpy as np
import pytest

from remodels.qra import SQRM

from . import sample_data


def test_sqrm_predicitons(sample_data: Tuple[np.array, np.array]) -> None:
    """Sum of SQRM model predictions equals specific number."""
    X_, y_ = sample_data
    assert SQRM(0.5, 1, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        991.161454057788, abs=1e-2
    )
    assert SQRM(0.5, None, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        990.1839456841076, abs=1e-2
    )
