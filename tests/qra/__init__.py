"""Test cases for the qra module."""
from typing import Tuple

import numpy as np
import pytest


@pytest.fixture
def sample_data() -> Tuple[np.array, np.array]:
    """Fixture providing sample data to test QRA models."""
    np.random.seed(0)
    X_ = np.random.uniform(0, 10, size=(100, 2))
    y_ = X_[:, 0] + X_[:, 1] + np.random.normal(0, 1, size=(100,))
    return X_, y_
