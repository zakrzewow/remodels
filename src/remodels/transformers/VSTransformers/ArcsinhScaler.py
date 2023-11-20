"""ArcsinhScaler."""

import numpy as np

from remodels.transformers.BaseScaler import BaseScaler


class ArcsinhScaler(BaseScaler):
    """Scaler that applies an arcsinh transformation to the data."""

    def transform(self, X, y=None):
        """Apply the arcsinh transformation to the features and optionally the target.

        :param X: Features to transform.
        :param y: Optional target to transform.
        :return: Transformed features and optionally transformed target.
        """
        X_transformed = np.arcsinh(X)
        y_transformed = np.arcsinh(y) if y is not None else None
        return (X_transformed, y_transformed) if y is not None else X_transformed

    def inverse_transform(self, X, y=None):
        """Apply the inverse arcsinh transformation to the features and optionally the target.

        :param X: Transformed features to inverse transform.
        :param y: Transformed target to inverse transform.
        :return: Inverse transformed features and optionally inverse transformed target.
        """
        X_inverted = np.sinh(X) if X is not None else None
        y_inverted = np.sinh(y) if y is not None else None
        return (X_inverted, y_inverted)
