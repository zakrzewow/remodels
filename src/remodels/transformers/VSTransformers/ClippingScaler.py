"""ClippingScaler."""

import numpy as np
import pandas as pd

from remodels.transformers.BaseScaler import BaseScaler


class ClippingScaler(BaseScaler):
    """Scaler that clips feature and target values to within k standard deviations from the mean."""

    def __init__(self, k=3):
        """Initialize the ClippingScaler with a clipping threshold.

        :param k: The number of standard deviations to use as the clipping threshold.
        :type k: int or float
        """
        self.k = k

    def fit(self, X, y=None):
        """Fit the scaler to the data.

        This scaler does not learn anything from the data
        and hence the fit method is a placeholder that returns self.

        :param X: Features to fit.
        :type X: np.ndarray or pd.DataFrame
        :param y: Optional target to fit. Not used in this scaler.
        :type y: np.ndarray or pd.Series, optional
        :return: The fitted scaler.
        :rtype: ClippingScaler
        """
        # No fitting necessary for ClippingScaler, so just return self.
        return self

    def _clip_data(self, data):
        """Clip the data to within the threshold defined by self.k.

        :param data: The data to clip.
        :type data: np.ndarray or pd.Series
        :return: The clipped data.
        :rtype: np.ndarray
        """
        if isinstance(data, pd.Series):
            data = data.values
        condition = np.abs(data) > self.k
        return np.where(condition, self.k * np.sign(data), data)

    def transform(self, X, y=None):
        """Transform the features and optionally the target by clipping their values.

        :param X: Features to transform.
        :type X: np.ndarray or pd.DataFrame
        :param y: Optional target to transform.
        :type y: np.ndarray or pd.Series, optional
        :return: The transformed features and optionally the transformed target.
        :rtype: tuple
        """
        X_transformed = self._clip_data(X)
        y_transformed = self._clip_data(y) if y is not None else None
        return (
            (self._to_dataframe(X, X_transformed), self._to_dataframe(y, y_transformed))
            if y is not None
            else self._to_dataframe(X, X_transformed)
        )

    def inverse_transform(self, X=None, y=None):
        """Inverse transform the features and optionally the target by unclipping their values.

        This method assumes the original data was within the range [-k, k].

        :param X: Transformed features to inverse transform.
        :type X: np.ndarray or pd.DataFrame
        :param y: Transformed target to inverse transform.
        :type y: np.ndarray or pd.Series, optional
        :return: The original features and target.
        :rtype: tuple
        """
        X_inverted = np.clip(X, -self.k, self.k) if X is not None else None
        y_inverted = np.clip(y, -self.k, self.k) if y is not None else None
        return (self._to_dataframe(X, X_inverted), self._to_dataframe(y, y_inverted))
