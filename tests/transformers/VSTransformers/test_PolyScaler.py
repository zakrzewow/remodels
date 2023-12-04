"""Test cases for the PolyScaler."""
import numpy as np
import pandas as pd
import pytest

from remodels.transformers.VSTransformers.PolyScaler import PolyScaler

from . import sample_dfs


def test_poly_scaler_transform(sample_dfs):
    """Test PolyScaler for correct polynomial transformation."""
    X_df, y_df = sample_dfs
    scaler = PolyScaler(lamb=0.125, c=0.05)

    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    c_lamb = (0.05 / 0.125) ** (1 / (0.125 - 1))
    expected_X = np.sign(X_df) * ((np.abs(X_df) + c_lamb) ** 0.125 - c_lamb ** (0.125))
    expected_y = np.sign(y_df) * ((np.abs(y_df) + c_lamb) ** 0.125 - c_lamb ** (0.125))

    assert np.allclose(X_transformed.values, expected_X, atol=1e-2)
    assert np.allclose(y_transformed.values, expected_y, atol=1e-2)


def test_poly_scaler_inverse_transform(sample_dfs):
    """Check PolyScaler's inverse transform restores original data."""
    X_df, y_df = sample_dfs
    scaler = PolyScaler(lamb=0.125, c=0.05)

    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)

    assert np.allclose(X_inverted, X_df, atol=1e-2)
    assert np.allclose(y_inverted, y_df, atol=1e-2)


def test_poly_scaler_output_types(sample_dfs):
    """Ensure PolyScaler's output types are DataFrames."""
    X_df, y_df = sample_dfs
    scaler = PolyScaler(lamb=0.125, c=0.05)

    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    assert isinstance(X_transformed, pd.DataFrame)
    assert isinstance(y_transformed, pd.DataFrame)

    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)
    assert isinstance(X_inverted, pd.DataFrame)
    assert isinstance(y_inverted, pd.DataFrame)
