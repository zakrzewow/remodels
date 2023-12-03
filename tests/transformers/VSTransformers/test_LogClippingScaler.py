"""Test cases for the LogClippingScaler."""
import numpy as np
import pandas as pd
import pytest

from remodels.transformers.VSTransformers.LogClippingScaler import LogClippingScaler

from . import sample_dfs


def test_logclipping_scaler_transform(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = LogClippingScaler(k=3)

    X_transformed, y_transformed = scaler.transform(X_df, y_df)

    # Creating a condition for values exceeding the threshold in the original data
    condition_X = np.abs(X_df) > 3
    condition_y = np.abs(y_df) > 3

    # Calculating expected transformed values using the log clipping logic
    expected_X = np.where(
        condition_X, np.sign(X_df) * (np.log(np.abs(X_df) - 2) + 3), X_df
    )
    expected_y = np.where(
        condition_y, np.sign(y_df) * (np.log(np.abs(y_df) - 2) + 3), y_df
    )

    assert np.allclose(X_transformed, expected_X)
    assert np.allclose(y_transformed, expected_y)


def test_logclipping_scaler_inverse_transform(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = LogClippingScaler(k=3)

    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    X_df[X_df <= 0] = np.nextafter(0, 1)
    y_df[y_df <= 0] = np.nextafter(0, 1)

    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)

    assert np.allclose(X_inverted, X_df, atol=1e-2)
    assert np.allclose(y_inverted, y_df, atol=1e-2)


def test_logclipping_scaler_output_types(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = LogClippingScaler(k=3)

    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    assert isinstance(X_transformed, pd.DataFrame)
    assert isinstance(y_transformed, pd.DataFrame)

    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)
    assert isinstance(X_inverted, pd.DataFrame)
    assert isinstance(y_inverted, pd.DataFrame)
