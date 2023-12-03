"""Test cases for the ArcsinhScaler."""
import numpy as np
import pandas as pd
import pytest

from remodels.transformers.VSTransformers.ArcsinhScaler import ArcsinhScaler

from . import sample_dfs


def test_arcsinh_scaler_transform_with_dataframe(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = ArcsinhScaler()

    X_transformed, y_transformed = scaler.fit_transform(X_df, y_df)
    assert np.allclose(X_transformed, np.arcsinh(X_df))
    assert np.allclose(y_transformed, np.arcsinh(y_df))


def test_arcsinh_scaler_inverse_transform_with_dataframe(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = ArcsinhScaler()

    X_transformed, y_transformed = scaler.fit_transform(X_df, y_df)
    X_inverted, y_transformed = scaler.inverse_transform(X_transformed, y_transformed)
    assert np.allclose(X_inverted, X_df)
    assert np.allclose(y_transformed, y_df)


def test_arcsinh_scaler_output_types(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = ArcsinhScaler()

    X_transformed, y_transformed = scaler.fit_transform(X_df, y_df)

    assert isinstance(X_transformed, pd.DataFrame)
    assert isinstance(y_transformed, pd.DataFrame)

    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)

    assert isinstance(X_inverted, pd.DataFrame)
    assert isinstance(y_inverted, pd.DataFrame)
