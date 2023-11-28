"""Test cases for the transformers module."""
from typing import Tuple

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_dfs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fixture providing sample data to test QRA models with a 7-day datetime index, hourly intervals."""
    np.random.seed(0)

    num_hours = 14 * 24
    X_ = np.random.uniform(0, 10, size=(num_hours, 2))
    y_ = X_[:, 0] + X_[:, 1] + np.random.normal(0, 1, size=num_hours)

    date_index = pd.date_range(start="2023-01-01", periods=num_hours, freq="H")

    X_df = pd.DataFrame(X_, index=date_index, columns=["feature1", "feature2"])
    y_df = pd.DataFrame(y_, index=date_index, columns=["target"])

    return X_df, y_df
