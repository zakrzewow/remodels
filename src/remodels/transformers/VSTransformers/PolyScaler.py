"""PolyScaler."""

from typing import Tuple

import numpy as np
import pandas as pd

from remodels.transformers.BaseScaler import BaseScaler


class PolyScaler(BaseScaler):
    r"""Scaler that applies a polynomial transformation to data, transforming each feature according to a polynomial function.

    The transformation is defined as:

        sign(x) * ((\|x\| + (c / lamb)^(1 / (lamb - 1)))^lamb - (c / lamb)^(lamb / (lamb - 1)))

    where 'lamb' is the exponent parameter, and 'c' is a constant determining the curvature of
    the polynomial. This transformation can be particularly useful for stabilizing variance and
    making skewed distributions more symmetric.

    The scaler also provides an inverse transformation function to revert the data back to
    its original scale.
    """

    def __init__(self, lamb: float = 0.125, c: float = 0.05) -> None:
        """Initialize the scaler with parameters for the polynomial transformation.

        :param lamb: Exponent used in the polynomial transformation.
        :type lamb: float
        :param c: Constant that defines the curvature of the polynomial transformation.
        :type c: float

        """
        self.lamb = lamb
        self.c = c

    def _transform_data(self, data: np.ndarray) -> np.ndarray:
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
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Inverse transform the features and optionally the target.

        :param X: Transformed features to inverse transform.
        :type X: np.ndarray
        :param y: Transformed target to inverse transform.
        :type y: np.ndarray, optional
        :return: Original features and optionally original target.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]

        """

        def invert(data):
            c_lamb = (self.c / self.lamb) ** (self.lamb / (self.lamb - 1))
            return np.sign(data) * (
                (np.abs(data) + c_lamb) ** (1 / self.lamb) - c_lamb ** (1 / (self.lamb))
            )

        X_inverted = invert(X) if X is not None else None
        y_inverted = invert(y) if y is not None else None
        return (X_inverted, y_inverted)
