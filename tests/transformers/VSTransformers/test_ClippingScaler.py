"""Test cases for the ClippingScaler."""
import numpy as np
import pandas as pd
import pytest

from remodels.transformers.VSTransformers.ClippingScaler import ClippingScaler

from . import sample_dfs


def test_clipping_scaler_transform(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = ClippingScaler(k=3)

    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    assert np.all(X_transformed <= 3)
    assert np.all(y_transformed <= 3)


def test_clipping_scaler_inverse_transform(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = ClippingScaler(k=3)

    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)
    assert np.allclose(X_inverted, X_transformed)
    assert np.allclose(y_inverted, y_transformed)


def test_clipping_scaler_output_types(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = ClippingScaler(k=3)

    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    assert isinstance(X_transformed, pd.DataFrame)
    assert isinstance(y_transformed, pd.DataFrame)
