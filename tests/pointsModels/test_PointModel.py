import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from remodels.pipelines.RePipeline import RePipeline
from remodels.pointsModels.PointModel import PointModel
from remodels.transformers.BaseScaler import BaseScaler

from . import sample_dfs


class MockTransformer(BaseScaler):
    def transform(self, X, y=None):
        return (X, y) if y is not None else X

    def inverse_transform(self, X=None, y=None):
        return X, y


def test_pointmodel_initialization(sample_dfs):
    X_df, y_df = sample_dfs
    pipeline = RePipeline(
        steps=[("mock", MockTransformer()), ("model", LinearRegression())]
    )
    variables_per_hour = {(0, 23): X_df.columns.tolist()}

    model = PointModel(
        pipeline=pipeline,
        variables_per_hour=variables_per_hour,
        y_column=y_df.columns[0],
    )

    assert isinstance(model, PointModel)


def test_pointmodel_fit(sample_dfs):
    X_df, y_df = sample_dfs
    pipeline = RePipeline(
        steps=[("mock", MockTransformer()), ("model", LinearRegression())]
    )
    variables_per_hour = {(0, 23): X_df.columns.tolist()}

    model = PointModel(
        pipeline=pipeline,
        variables_per_hour=variables_per_hour,
        y_column=y_df.columns[0],
    )

    model.fit(X_df.join(y_df), start="2023-01-07", end="2023-01-14")
    assert model.training_data is not None


def test_pointmodel_predict(sample_dfs):
    X_df, y_df = sample_dfs
    pipeline = RePipeline(
        steps=[("mock", MockTransformer()), ("model", LinearRegression())]
    )
    variables_per_hour = {(0, 23): X_df.columns.tolist()}

    model = PointModel(
        pipeline=pipeline,
        variables_per_hour=variables_per_hour,
        y_column=y_df.columns[0],
    )

    model.fit(X_df.join(y_df), start="2023-01-10", end="2023-01-14")
    print(X_df.join(y_df))
    predictions = model.predict(rolling_window=5, inverse_predictions=True)
    assert isinstance(predictions, pd.DataFrame)


def test_pointmodel_summary(sample_dfs):
    X_df, y_df = sample_dfs
    pipeline = RePipeline(
        steps=[("mock", MockTransformer()), ("model", LinearRegression())]
    )
    variables_per_hour = {(0, 24): X_df.columns.tolist()}

    model = PointModel(
        pipeline=pipeline,
        variables_per_hour=variables_per_hour,
        y_column=y_df.columns[0],
    )

    model.fit(X_df.join(y_df), start="2023-01-10", end="2023-01-14")
    model.predict(rolling_window=5, inverse_predictions=True)
    summary = model.summary()
    assert isinstance(summary, pd.DataFrame)
