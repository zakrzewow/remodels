"""Test cases for the FQRA model."""
from typing import Tuple

import numpy as np
import pytest

from remodels.qra import FQRA

from . import sample_data


def test_fqra_predicitons(sample_data: Tuple[np.array, np.array]) -> None:
    """Sum of FQRA model predictions equals specific number."""
    X_, y_ = sample_data
    assert FQRA(0.5, 1, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        987.7472734112102, abs=1e-2
    )
    assert FQRA(0.5, 1, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        FQRA(0.5, None, True).fit(X_, y_).predict(X_).sum()
    )
    assert FQRA(0.5, 2, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        984.583331019932, abs=1e-2
    )
