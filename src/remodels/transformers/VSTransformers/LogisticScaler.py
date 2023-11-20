"""LogisticScaler."""

import numpy as np

from remodels.transformers.BaseScaler import BaseScaler


class LogisticScaler(BaseScaler):
    """Scaler that applies a logistic transformation to the data."""

    def transform(self, X, y=None):
        """Apply the logistic transformation to the features and optionally the target.

        :param X: Features to transform.
        :param y: Optional target to transform.
        :return: Transformed features and optionally transformed target.
        """
        X_transformed = (1 + np.exp(-X)) ** (-1)
        y_transformed = (1 + np.exp(-y)) ** (-1) if y is not None else None
        return (X_transformed, y_transformed) if y is not None else X_transformed

    def inverse_transform(self, X=None, y=None):
        """Apply the inverse logistic transformation to the features and optionally the target.

        :param X: Transformed features to inverse transform.
        :param y: Transformed target to inverse transform.
        :return: Inverse transformed features and optionally inverse transformed target.
        """

        def invert(data):
            return np.log(data / (1 - data))

        X_inverted = invert(X) if X is not None else None
        y_inverted = invert(y) if y is not None else None
        return (X_inverted, y_inverted)
