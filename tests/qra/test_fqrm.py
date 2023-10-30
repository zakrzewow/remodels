"""Test cases for the FQRM model."""
from typing import Tuple

import numpy as np
import pytest

from remodels.qra import FQRM

from . import sample_data


def test_fqrm_predicitons(sample_data: Tuple[np.array, np.array]) -> None:
    """Sum of FQRM model predictions equals specific number."""
    X_, y_ = sample_data
    assert FQRM(0.5, 1, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        987.7472734112102, abs=1e-2
    )
    assert FQRM(0.5, 1, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        FQRM(0.5, None, True).fit(X_, y_).predict(X_).sum()
    )
    assert FQRM(0.5, 2, True).fit(X_, y_).predict(X_).sum() == pytest.approx(
        981.7977917308983, abs=1e-2
    )
