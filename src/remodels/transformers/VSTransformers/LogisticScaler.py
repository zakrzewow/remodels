"""LogisticScaler."""

from typing import Tuple

import numpy as np
import pandas as pd

from remodels.transformers.BaseScaler import BaseScaler


class LogisticScaler(BaseScaler):
    """Scaler that applies a logistic transformation to the data. This transformation converts each feature using the logistic function, which maps any real-valued number into the range (0, 1).

    The transformation is particularly useful in preparing data for algorithms that expect input values
    to be in a bounded range. It can also help in dealing with features that have skewed distributions.

    The logistic transformation is defined as:
        1 / (1 + exp(-x))

    where 'x' is the feature value.

    The scaler provides both the transformation and its inverse, allowing the original scale of the data
    to be recovered.

    The scaler also provides an inverse transformation function to revert the data back to
    its original scale.
    """

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the logistic transformation to the features and optionally the target.

        :param X: Features to transform.
        :type X: pd.DataFrame
        :param y: Optional target to transform.
        :type y: pd.DataFrame, optional
        :return: Transformed features and optionally transformed target.
        :rtype: pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        """
        X_transformed = (1 + np.exp(-X)) ** (-1)
        y_transformed = (1 + np.exp(-y)) ** (-1) if y is not None else None
        return (X_transformed, y_transformed) if y is not None else X_transformed

    def inverse_transform(
        self, X: pd.DataFrame = None, y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the inverse logistic transformation to the features and optionally the target.

        :param X: Transformed features to inverse transform.
        :type X: pd.DataFrame
        :param y: Transformed target to inverse transform.
        :type y: pd.DataFrame, optional
        :return: Original features and optionally original target.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """

        def invert(data):
            return np.log(data / (1 - data))

        X_inverted = invert(X) if X is not None else None
        y_inverted = invert(y) if y is not None else None
        return (X_inverted, y_inverted)
