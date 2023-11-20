"""MLogScaler."""

import numpy as np

from remodels.transformers.BaseScaler import BaseScaler


class MLogScaler(BaseScaler):
    """Scaler that applies a modified logarithmic transformation to the data."""

    def __init__(self, c=1 / 3):
        """Initialize the scaler with a constant used in the transformation.

        :param c: A small constant to ensure non-zero division in transformation.
        :type c: float
        """
        self.c = c

    def _transform_data(self, data):
        """Apply the modified logarithmic transformation to the data.

        :param data: Data to transform.
        :type data: np.ndarray
        :return: Transformed data.
        :rtype: np.ndarray
        """
        return np.sign(data) * (np.log(np.abs(data) + 1 / self.c) + np.log(self.c))

    def transform(self, X, y=None):
        """Transform the features and optionally the target.

        :param X: Features to transform.
        :type X: np.ndarray
        :param y: Optional target to transform.
        :type y: np.ndarray, optional
        :return: Transformed features and optionally transformed target.
        :rtype: tuple
        """
        X_transformed = self._transform_data(X)
        return (
            (X_transformed, self._transform_data(y)) if y is not None else X_transformed
        )

    def inverse_transform(self, X=None, y=None):
        """Inverse transform the features and optionally the target.

        :param X: Transformed features to inverse transform.
        :type X: np.ndarray
        :param y: Transformed target to inverse transform.
        :type y: np.ndarray, optional
        :return: Original features and optionally original target.
        :rtype: tuple
        """

        def invert(data):
            return np.sign(data) * (np.exp(np.abs(data) - np.log(self.c)) - 1 / self.c)

        X_inverted = invert(X) if X is not None else None
        y_inverted = invert(y) if y is not None else None
        return (self._to_dataframe(X, X_inverted), self._to_dataframe(y, y_inverted))
