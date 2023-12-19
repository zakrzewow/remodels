"""BoxCoxScaler."""

from typing import Tuple

import numpy as np
import pandas as pd

from remodels.transformers.BaseScaler import BaseScaler


class BoxCoxScaler(BaseScaler):
    """A scaler that applies a Box-Cox transformation to the data.

    The Box-Cox transformation is a statistical technique used to stabilize variance, make the data
    more normally distributed, and improve the validity of measures of association. It's particularly
    effective for transforming non-normal dependent variables into a normal shape. The transformation
    is defined as:

    Y(λ) = (X^λ - 1) / λ, if λ != 0
           log(X), if λ = 0

    where X is the original data and λ is the transformation parameter. The λ value is chosen to
    maximize the normality of the transformed data. A λ of 0 implies a log transformation, while
    other values indicate various degrees of exponential transformation.

    This scaler includes both the Box-Cox transformation and its inverse, enabling reversible scaling
    of data.
    """

    def __init__(self, lamb: float = 0.5) -> None:
        """Initialize the scaler with a lambda parameter for the Box-Cox transformation.

        :param lamb: Lambda parameter for the Box-Cox transformation.
        :type lamb: float
        """
        self.lamb = lamb

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the Box-Cox transformation to the data.

        :param data: Data to transform.
        :type data: pd.DataFrame or np.ndarray
        :return: Transformed data.
        :rtype: pd.DataFrame
        """
        if self.lamb != 0:
            transformed = np.sign(data) * (
                ((np.abs(data) + 1) ** self.lamb - 1) / self.lamb
            )
        else:
            transformed = np.log(np.abs(data) + 1)
        return transformed

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the transformation to the features and optionally the target.

        :param X: Features to transform.
        :type X: pd.DataFrame
        :param y: Optional target to transform.
        :type y: pd.DataFrame
        :return: Transformed features and optionally transformed target.
        :rtype: pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        """
        X_transformed = self._transform_data(X)
        y_transformed = self._transform_data(y) if y is not None else None
        return (X_transformed, y_transformed) if y is not None else X_transformed

    def inverse_transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the inverse Box-Cox transformation to the features and optionally the target.

        :param X: Transformed features to inverse transform.
        :type X: pd.DataFrame
        :param y: Transformed target to inverse transform.
        :type y: pd.DataFrame
        :return: Inverse transformed features and optionally inverse transformed target.
        :rtype: pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        """

        def invert(data):
            return (
                np.sign(data) * ((self.lamb * np.abs(data) + 1) ** (1 / self.lamb) - 1)
                if self.lamb != 0
                else np.exp(np.abs(data)) - 1
            )

        X_inverted = invert(X) if X is not None else None
        y_inverted = invert(y) if y is not None else None
        return (X_inverted, y_inverted)
