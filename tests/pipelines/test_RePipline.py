"""Test cases for the PITScaler."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from remodels.pipelines.RePipeline import RePipeline

from . import sample_dfs


class MockTransformer(TransformerMixin, BaseEstimator):
    """Mock transformer for testing."""

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        return X * 2, y if y is not None else X * 2

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        return X / 2, y if y is not None else X / 2


def test_repipeline_fit(sample_dfs):
    X, y = sample_dfs
    pipeline = RePipeline(steps=[("mock", MockTransformer())])

    fitted_pipeline = pipeline.fit(X, y)
    assert isinstance(fitted_pipeline, RePipeline)


def test_repipeline_transform(sample_dfs):
    X, y = sample_dfs
    pipeline = RePipeline(steps=[("mock", MockTransformer())])

    pipeline.fit(X, y)
    Xt, yt = pipeline.transform(X, y)
    assert np.array_equal(Xt, X * 2)
    assert yt is None or np.array_equal(yt, y)


def test_repipeline_fit_transform(sample_dfs):
    X, y = sample_dfs
    pipeline = RePipeline(steps=[("mock", MockTransformer())])

    Xt, yt = pipeline.fit_transform(X, y)

    assert np.allclose(Xt, X * 2)
    assert yt is None or np.allclose(yt, y)


def test_repipeline_inverse_transform(sample_dfs):
    X, y = sample_dfs
    pipeline = RePipeline(steps=[("mock", MockTransformer())])

    pipeline.fit(X, y)
    Xt, yt = pipeline.transform(X, y)
    X_inv, y_inv = pipeline.inverse_transform(Xt, yt)
    assert np.array_equal(X_inv, X)
    assert y_inv is None or np.array_equal(y_inv, y)
