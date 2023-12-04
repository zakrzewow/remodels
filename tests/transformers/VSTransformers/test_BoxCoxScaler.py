"""Test cases for the BoxCoxScaler."""
import numpy as np
import pandas as pd
import pytest

from remodels.transformers.VSTransformers.BoxCoxScaler import BoxCoxScaler

from . import sample_dfs


def test_boxcox_scaler_output_types(sample_dfs):
    """Ensure BoxCoxScaler's transform and inverse_transform return DataFrames."""
    X_df, y_df = sample_dfs
    scaler = BoxCoxScaler(lamb=0.5)

    X_transformed, y_transformed = scaler.fit_transform(X_df, y_df)
    assert isinstance(X_transformed, pd.DataFrame)
    assert isinstance(y_transformed, pd.DataFrame)

    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)
    assert isinstance(X_inverted, pd.DataFrame)
    assert isinstance(y_inverted, pd.DataFrame)


def test_boxcox_scaler_known_values():
    """Test BoxCoxScaler's transformation and inversion with known values."""
    scaler = BoxCoxScaler(lamb=0.5)
    X = pd.DataFrame([[0.1, 1], [2, 3]], columns=["feature1", "feature2"])
    y = pd.DataFrame([[0.1], [4]], columns=["target"])

    # Apply BoxCox transformation
    X_transformed, y_transformed = scaler.transform(X, y)

    expected_X = np.sign(X) * (((np.abs(X) + 1) ** 0.5 - 1) / 0.5)
    expected_y = np.sign(y) * (((np.abs(y) + 1) ** 0.5 - 1) / 0.5)

    assert np.allclose(X_transformed, expected_X)
    assert np.allclose(y_transformed, expected_y)

    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)
    assert np.allclose(X_inverted, X)
    assert np.allclose(y_inverted, y)
