"""ArcsinhScaler."""

from typing import Tuple

import numpy as np
import pandas as pd

from remodels.transformers.BaseScaler import BaseScaler


class ArcsinhScaler(BaseScaler):
    """A scaler that applies an arcsinh (inverse hyperbolic sine) transformation to the data.

    This scaler is useful for handling data with skewed distributions and can help in stabilizing
    the variance of the data. The transformation is stateless and does not depend on the data itself,
    meaning no fitting is required.

    The scaler also provides an inverse transformation function, which applies the sinh (hyperbolic sine)
    transformation to revert the data back to its original scale.
    """

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the arcsinh transformation to the features and optionally the target.

        :param X: Features to transform.
        :type X: pd.DataFrame
        :param y: Optional target to transform.
        :type y: pd.DataFrame
        :return: Transformed features and optionally transformed target.
        :rtype: pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        """
        X_transformed = np.arcsinh(X)
        y_transformed = np.arcsinh(y) if y is not None else None
        return (X_transformed, y_transformed) if y is not None else X_transformed

    def inverse_transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the inverse arcsinh transformation to the features and optionally the target.

        :param X: Transformed features to inverse transform.
        :type X: pd.DataFrame
        :param y: Transformed target to inverse transform.
        :type y: pd.DataFrame
        :return: Inverse transformed features and optionally inverse transformed target.
        :type: Tuple[pd.DataFrame, pd.DataFrame]
        """
        X_inverted = np.sinh(X) if X is not None else None
        y_inverted = np.sinh(y) if y is not None else None
        return (X_inverted, y_inverted)
