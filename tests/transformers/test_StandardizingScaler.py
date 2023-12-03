"""Test cases for the StandardizingScaler."""
import numpy as np
import pandas as pd
import pytest

from remodels.transformers.StandardizingScaler import StandardizingScaler

from . import sample_dfs


def test_standardizing_scaler_mean_method(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = StandardizingScaler(method="mean")

    scaler.fit(X_df, y_df)
    X_transformed, y_transformed = scaler.transform(X_df, y_df)

    # Check if the mean of the transformed data is approximately zero
    assert X_transformed.mean().mean() == pytest.approx(0, abs=1e-2)
    assert y_transformed.mean().mean() == pytest.approx(0, abs=1e-2)


def test_standardizing_scaler_median_method(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = StandardizingScaler(method="median")

    scaler.fit(X_df, y_df)
    X_transformed, y_transformed = scaler.transform(X_df, y_df)

    # Check if the median of the transformed data is approximately zero
    assert X_transformed.median().median() == pytest.approx(0, abs=1e-2)
    assert y_transformed.median().median() == pytest.approx(0, abs=1e-2)


def test_standardizing_scaler_inverse_transform(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = StandardizingScaler(method="mean")

    scaler.fit(X_df, y_df)
    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)

    # Check if the inverse transformed data matches the original
    assert np.allclose(X_inverted.to_numpy(), X_df.to_numpy())
    assert np.allclose(y_inverted.to_numpy(), y_df.to_numpy())


def test_standardizing_scaler_with_constant_values(sample_dfs):
    X_df, y_df = sample_dfs
    # Replace X_df with constant values
    X_df_constant = X_df.copy()
    X_df_constant["feature1"] = 5
    X_df_constant["feature2"] = 10
    y_df["target"] = 10

    scaler = StandardizingScaler(method="mean")
    scaler.fit(X_df_constant, y_df)
    X_transformed, y_transformed = scaler.transform(X_df_constant, y_df)

    # Check if constant values are handled correctly (transformed to 0)
    assert X_transformed["feature1"].mean().mean() == pytest.approx(0, abs=1e-2)
    assert X_transformed["feature2"].mean().mean() == pytest.approx(0, abs=1e-2)
    assert y_transformed.mean().mean() == pytest.approx(0, abs=1e-2)


def test_standardizing_scaler_empty_dataframe():
    X_df_empty = pd.DataFrame()
    scaler = StandardizingScaler(method="mean")

    with pytest.raises(ValueError):
        scaler.fit(X_df_empty)


def test_standardizing_scaler_invalid_method():
    with pytest.raises(ValueError):
        StandardizingScaler(method="invalid_method")


def test_standardizing_scaler_transform_output_type(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = StandardizingScaler(method="mean")

    scaler.fit(X_df, y_df)
    X_transformed, y_transformed = scaler.transform(X_df, y_df)

    # Check if the transformed features and target are DataFrames and Series respectively
    assert isinstance(X_transformed, pd.DataFrame)
    assert isinstance(y_transformed, pd.DataFrame)


def test_standardizing_scaler_inverse_transform_output_type(sample_dfs):
    X_df, y_df = sample_dfs
    scaler = StandardizingScaler(method="mean")

    scaler.fit(X_df, y_df)
    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)

    # Check if the inverse transformed features and target are DataFrames and Series respectively
    assert isinstance(X_inverted, pd.DataFrame)
    assert isinstance(y_inverted, pd.DataFrame)
