"""TimeTransformers."""

import pandas as pd

from remodels.transformers.BaseScaler import BaseScaler


class DSTAdjuster(BaseScaler):
    """Adjusts time series data for Daylight Saving Time changes."""

    def __init__(self):
        """Initialize the DSTAdjuster."""
        super().__init__()

    def fit(self, X, y=None):
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

    def transform(self, X: pd.DataFrame, y=None):
        """Transform the time series data to adjust for DST changes.

        :param X: Time series data to transform.
        :type X: pd.DataFrame
        :param y: Optional target series corresponding to the time series data.
        :type y: pd.Series, optional
        :return: Adjusted time series data, and optionally the target series.
        :rtype: pd.DataFrame, pd.Series
        """
        # Remove timezone information and resample to hourly frequency.
        X_adj = X.tz_localize(None).resample("H").mean()

        # Fill missing values by averaging the adjacent values, then forward fill.
        X_adj = X_adj.fillna(method="bfill").fillna(method="ffill")

        if y is not None:
            # Ensure y is a pandas Series with the correct index.
            y_adj = pd.Series(y, index=X_adj.index)

            # Fill missing values in y using the same approach as for X.
            y_adj = y_adj.fillna(method="bfill").fillna(method="ffill")

            return X_adj, y_adj

        return X_adj
