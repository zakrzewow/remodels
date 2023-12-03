"""Test cases for the LogisticScaler."""
import numpy as np
import pandas as pd
import pytest

from remodels.transformers.VSTransformers.LogisticScaler import LogisticScaler

from . import sample_dfs


def test_logistic_scaler_transform(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = LogisticScaler()

    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    expected_X = (1 + np.exp(-X_df)) ** (-1)
    expected_y = (1 + np.exp(-y_df)) ** (-1)

    assert np.allclose(X_transformed, expected_X)
    assert np.allclose(y_transformed, expected_y)


def test_logistic_scaler_inverse_transform(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = LogisticScaler()

    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)

    assert np.allclose(X_inverted, X_df, atol=1e-2)
    assert np.allclose(y_inverted, y_df, atol=1e-2)


def test_logistic_scaler_output_types(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = LogisticScaler()

    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    assert isinstance(X_transformed, pd.DataFrame)
    assert isinstance(y_transformed, pd.DataFrame)

    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)
    assert isinstance(X_inverted, pd.DataFrame)
    assert isinstance(y_inverted, pd.DataFrame)
