"""MLogScaler."""

from typing import Tuple

import numpy as np
import pandas as pd

from remodels.transformers.BaseScaler import BaseScaler


class MLogScaler(BaseScaler):
    r"""Scaler that applies a modified logarithmic transformation to the data. This transformation is designed to handle zero and negative values effectively by incorporating a small constant.

    The transformation is defined as:
        sign(x) * (log(\|x\| + 1/c) + log(c))

    where 'c' is a small constant to ensure non-zero division. This transformation helps in
    stabilizing variance and normalizing distributions, especially useful for skewed data.

    The scaler also provides an inverse transformation function to revert the data back to
    its original scale.
    """

    def __init__(self, c: int = 1 / 3):
        """Initialize the scaler with a constant used in the transformation.

        :param c: A small constant to ensure non-zero division in transformation.
        :type c: float
        """
        self.c = c

    def _transform_data(self, data: pd.DataFrame):
        """Apply the modified logarithmic transformation to the data.

        :param data: Data to transform.
        :type data: pd.DataFrame
        :return: Transformed data.
        :rtype: pd.DataFrame
        """
        return np.sign(data) * (np.log(np.abs(data) + 1 / self.c) + np.log(self.c))

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform the features and optionally the target.

        :param X: Features to transform.
        :type X: pd.DataFrame
        :param y: Optional target to transform.
        :type y: pd.DataFrame, optional
        :return: Transformed features and optionally transformed target.
        :rtype: pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        """
        X_transformed = self._transform_data(X)
        return (
            (X_transformed, self._transform_data(y)) if y is not None else X_transformed
        )

    def inverse_transform(
        self, X: pd.DataFrame = None, y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Inverse transform the features and optionally the target.

        :param X: Transformed features to inverse transform.
        :type X: pd.DataFrame
        :param y: Transformed target to inverse transform.
        :type y: pd.DataFrame, optional
        :return: Original features and optionally original target.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """

        def invert(data):
            return np.sign(data) * (np.exp(np.abs(data) - np.log(self.c)) - 1 / self.c)

        X_inverted = invert(X) if X is not None else None
        y_inverted = invert(y) if y is not None else None
        return (self._to_dataframe(X, X_inverted), self._to_dataframe(y, y_inverted))
