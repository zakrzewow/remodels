"""BoxCoxScaler."""

import numpy as np

from remodels.transformers.BaseScaler import BaseScaler


class BoxCoxScaler(BaseScaler):
    """Scaler that applies a Box-Cox transformation to the data."""

    def __init__(self, lamb=0.5):
        """Initialize the scaler with a lambda parameter for the Box-Cox transformation.

        :param lamb: Lambda parameter for the Box-Cox transformation.
        :type lamb: float
        """
        self.lamb = lamb

    def _transform_data(self, data):
        """Apply the Box-Cox transformation to the data.

        :param data: Data to transform.
        :type data: pd.DataFrame or np.ndarray
        :return: Transformed data.
        :rtype: pd.DataFrame or np.ndarray
        """
        if self.lamb != 0:
            transformed = np.sign(data) * (
                ((np.abs(data) + 1) ** self.lamb - 1) / self.lamb
            )
        else:
            transformed = np.log(np.abs(data) + 1)
        return transformed

    def transform(self, X, y=None):
        """Apply the transformation to the features and optionally the target.

        :param X: Features to transform.
        :param y: Optional target to transform.
        :return: Transformed features and optionally transformed target.
        """
        X_transformed = self._transform_data(X)
        y_transformed = self._transform_data(y) if y is not None else None
        return (X_transformed, y_transformed) if y is not None else X_transformed

    def inverse_transform(self, X, y=None):
        """Apply the inverse Box-Cox transformation to the features and optionally the target.

        :param X: Transformed features to inverse transform.
        :param y: Transformed target to inverse transform.
        :return: Inverse transformed features and optionally inverse transformed target.
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
