"""LogClippingScaler."""

from typing import Tuple

import numpy as np
import pandas as pd

from remodels.transformers.BaseScaler import BaseScaler


class LogClippingScaler(BaseScaler):
    """Scaler that applies a logarithmic transformation to values exceeding a specified threshold.

    This scaler is designed to transform features by applying a logarithmic transformation, but only to
    values that exceed a certain threshold, 'k'. This approach can be particularly useful in reducing the
    impact of outliers or extreme values in the data, while maintaining the scale of the rest of the data.

    The scaler also provides an inverse transformation function to revert the data back to
    its original scale.
    """

    def __init__(self, k=3):
        """Initialize the scaler with a clipping threshold.

        :param k: Clipping threshold.
        :type k: float
        """
        self.k = k

    def _transform_data(self, data):
        """Apply the log clipping transformation to the data.

        :param data: Data to transform.
        :type data: pd.DataFrame or np.ndarray
        :return: Transformed data.
        :rtype: pd.DataFrame or np.ndarray
        """
        condition = np.abs(data) > self.k
        data_transformed = data.copy()
        data_transformed = np.maximum(data_transformed.abs(), 2 + 1e-8)
        data_transformed = np.where(
            condition, np.sign(data) * (np.log(np.abs(data_transformed) - 2) + 3), data
        )
        return data_transformed

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the transformation to the features and optionally the target.

        :param X: Features to transform.
        :type X: pd.DataFrame
        :param y: Optional target to transform.
        :type y: pd.DataFrame, optional
        :return: Transformed features and optionally transformed target.
        :rtype: pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        """
        X_transformed = self._transform_data(X)
        y_transformed = self._transform_data(y) if y is not None else None
        return (
            (self._to_dataframe(X, X_transformed), self._to_dataframe(y, y_transformed))
            if y is not None
            else self._to_dataframe(X, X_transformed)
        )

    def inverse_transform(
        self, X: pd.DataFrame = None, y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the inverse log clipping transformation to the features and optionally the target.

        :param X: Transformed features to inverse transform.
        :type X: pd.DataFrame
        :param y: Transformed target to inverse transform.
        :type y: pd.DataFrame, optional
        :return: Original features and optionally original target.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """

        def invert(data):
            condition = np.abs(data) > self.k
            data_inverted = np.where(
                condition, np.sign(data) * (np.exp(np.abs(data) - 3) + 2), data
            )
            return data_inverted

        X_inverted = invert(X) if X is not None else None
        y_inverted = invert(y) if y is not None else None
        return (self._to_dataframe(X, X_inverted), self._to_dataframe(y, y_inverted))
