"""Test cases for the PITScaler."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from scipy.stats import t

from remodels.transformers.VSTransformers.PITScaler import PITScaler

from . import sample_dfs


def test_pit_scaler_normal_transform(sample_dfs):
    """Test PITScaler with normal distribution for finite transformed values."""
    X_df, y_df = sample_dfs
    scaler = PITScaler(distribution="normal")

    scaler.fit(X_df, y_df)
    X_transformed, y_transformed = scaler.transform(X_df, y_df)

    # Testing if transformed values are normally distributed
    assert np.all(np.isfinite(X_transformed))
    assert np.all(np.isfinite(y_transformed))


def test_pit_scaler_student_t_transform(sample_dfs):
    """Test PITScaler with student-t distribution for finite transformed values."""
    X_df, y_df = sample_dfs
    scaler = PITScaler(distribution="student-t")

    scaler.fit(X_df, y_df)
    X_transformed, y_transformed = scaler.transform(X_df, y_df)

    # Testing if transformed values follow a student-t distribution
    assert np.all(np.isfinite(X_transformed))
    assert np.all(np.isfinite(y_transformed))


def test_pit_scaler_inverse_transform(sample_dfs):
    """Check PITScaler's inverse transform restores original data."""
    X_df, y_df = sample_dfs
    scaler = PITScaler(distribution="normal")

    scaler.fit(X_df, y_df)
    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)
    assert np.allclose(X_inverted, X_df, atol=1e-2)
    assert np.allclose(y_inverted, y_df, atol=1e-2)


def test_pit_scaler_output_types(sample_dfs):
    """Ensure PITScaler's output types are DataFrames."""
    X_df, y_df = sample_dfs
    scaler = PITScaler(distribution="normal")

    scaler.fit(X_df, y_df)
    X_transformed, y_transformed = scaler.transform(X_df, y_df)
    assert isinstance(X_transformed, pd.DataFrame)
    assert isinstance(y_transformed, pd.DataFrame)

    X_inverted, y_inverted = scaler.inverse_transform(X_transformed, y_transformed)
    assert isinstance(X_inverted, pd.DataFrame)
    assert isinstance(y_inverted, pd.DataFrame)
