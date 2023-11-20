"""PolyScaler."""

import numpy as np

from remodels.transformers.BaseScaler import BaseScaler


class PolyScaler(BaseScaler):
    """Scaler that applies a polynomial transformation to the data."""

    def __init__(self, lamb=0.125, c=0.05):
        """Initialize the scaler with parameters for the polynomial transformation.

        :param lamb: Exponent used in the polynomial transformation.
        :type lamb: float
        :param c: Constant that defines the curvature of the polynomial transformation.
        :type c: float
        """
        self.lamb = lamb
        self.c = c

    def _transform_data(self, data):
        """Apply the polynomial transformation to the data.

        :param data: Data to transform.
        :type data: np.ndarray
        :return: Transformed data.
        :rtype: np.ndarray
        """
        return np.sign(data) * (
            (data.abs() + (self.c / self.lamb) ** (1 / (self.lamb - 1))).abs()
            ** self.lamb
            - (self.c / self.lamb) ** (self.lamb / (self.lamb - 1))
        )

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

    def inverse_transform(self, X, y=None):
        """Inverse transform the features and optionally the target.

        :param X: Transformed features to inverse transform.
        :type X: np.ndarray
        :param y: Transformed target to inverse transform.
        :type y: np.ndarray, optional
        :return: Original features and optionally original target.
        :rtype: tuple
        """

        def invert(data):
            c_lamb = (self.c / self.lamb) ** (self.lamb / (self.lamb - 1))
            return np.sign(data) * (
                (np.abs(data) + c_lamb) ** (1 / self.lamb) - c_lamb ** (1 / (self.lamb))
            )

        X_inverted = invert(X) if X is not None else None
        y_inverted = invert(y) if y is not None else None
        return (X_inverted, y_inverted)
