"""Test cases for the DSTAdjuster transformer."""
import pandas as pd
import pytest

from remodels.transformers.DSTAdjuster import DSTAdjuster


def test_dst_adjuster_spring_adjustment() -> None:
    """Test DSTAdjuster for correct handling of spring DST transition."""
    spring_rng = pd.date_range("2023-03-26 01:00:00", periods=4, freq="H", tz="CET")
    spring_X = pd.DataFrame({"value": [1, 3, 4, 5]}, index=spring_rng)
    dst_adjuster = DSTAdjuster()

    spring_X_transformed = dst_adjuster.transform(spring_X)
    spring_expected_values = [1, 2, 3, 4, 5]

    assert spring_X_transformed.index.tzinfo is None
    assert spring_X_transformed.isna().sum().sum() == 0
    assert all(spring_X_transformed["value"] == spring_expected_values)


def test_dst_adjuster_autumn_adjustment() -> None:
    """Test DSTAdjuster for correct handling of autumn DST transition."""
    autumn_rng = pd.date_range("2023-10-29 01:00:00", periods=5, freq="H", tz="CET")
    autumn_X = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=autumn_rng)
    dst_adjuster = DSTAdjuster()

    autumn_X_transformed = dst_adjuster.transform(autumn_X)
    autumn_expected_values = [1, 2.5, 4, 5]

    assert autumn_X_transformed.index.tzinfo is None
    assert autumn_X_transformed.isna().sum().sum() == 0
    assert all(autumn_X_transformed["value"] == autumn_expected_values)


def test_dst_adjuster_return_types() -> None:
    """Transformed output of DSTAdjuster is DataFrame."""
    spring_rng = pd.date_range("2023-03-26 01:00:00", periods=4, freq="H", tz="CET")
    spring_X = pd.DataFrame({"value": [1, 3, 4, 5]}, index=spring_rng)
    autumn_rng = pd.date_range("2023-10-29 01:00:00", periods=5, freq="H", tz="CET")
    autumn_X = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=autumn_rng)

    dst_adjuster = DSTAdjuster()

    spring_X_transformed = dst_adjuster.transform(spring_X)
    autumn_X_transformed = dst_adjuster.transform(autumn_X)

    assert isinstance(spring_X_transformed, pd.DataFrame)
    assert isinstance(autumn_X_transformed, pd.DataFrame)
