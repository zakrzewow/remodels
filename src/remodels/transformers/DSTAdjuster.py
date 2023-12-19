"""TimeTransformers."""

from typing import Tuple

import pandas as pd

from remodels.transformers.BaseScaler import BaseScaler


class DSTAdjuster(BaseScaler):
    """A transformer for adjusting time series data to account for Daylight Saving Time (DST) changes.

    This class provides functionality to modify time series data by removing timezone information and
    resampling to an hourly frequency. It's designed to handle potential issues arising from DST transitions,
    such as duplicate or missing timestamps. The transformer can be used with any time series data that
    includes timezone-aware datetime indices.
    """

    def __init__(self):
        """Initialize the DSTAdjuster."""
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "DSTAdjuster":
        """Fit the transformer to the data.

        This transformer does not learn anything from the data
        and hence the fit method is a placeholder that returns self.

        :param X: Features to fit.
        :type X: pd.DataFrame
        :param y: Optional target to fit. Not used in this transformer.
        :type y: pd.Series, optional
        :return: The fitted transformer.
        :rtype: DSTAdjuster
        """
        # No fitting necessary for DSTAdjuster, so just return self.
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform the time series data to adjust for DST changes.

        :param X: Time series data to trasnsform.
        :type X: pd.DataFrame
        :param y: Optional target series corresponding to the time series data.
        :type y: pd.Series, optional
        :return: Adjusted time series data, and optionally the target series.
        :rtype: pd.DataFrame, pd.Series
        """
        # Remove timezone information and resample to hourly frequency.
        X_adj = X.tz_localize(None).resample("H").mean()

        # Fill missing values by averaging the adjacent values, then forward fill.
        X_adj = X_adj.fillna(X_adj.shift(-1) / 2 + X_adj.shift(1) / 2)

        if y is not None:
            # Ensure y is a pandas Series with the correct index.
            y_adj = pd.Series(y, index=X_adj.index)

            # Fill missing values in y using the same approach as for X.
            y_adj = y_adj.fillna(y_adj.shift(-1) / 2 + y_adj.shift(1) / 2)

            return X_adj, y_adj

        return X_adj
