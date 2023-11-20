"""StandardizingScaler."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from remodels.transformers.BaseScaler import BaseScaler


def mad(x):
    """Compute the Mean Absolute Deviation (MAD) of an array."""
    median = np.median(x)
    dev = np.abs(x - median)
    return np.mean(dev)


class StandardizingScaler(BaseScaler):
    """A custom scaler for standardizing data using either the median or mean method."""

    def __init__(self, method="median"):
        """Initialize the StandardizingScaler with the chosen method of centering and scaling.

        :param method: Method to use for centering ('median' or 'mean').
        :type method: str
        """
        if method not in ["median", "mean"]:
            raise ValueError("Method must be 'median' or 'mean'")
        self.method = method
        self.x_centers = None
        self.x_scales = None
        self.y_center = None
        self.y_scale = None

    def _compute_center_scale(self, data):
        """Compute the center and scale of the data based on the specified method.

        :param data: The data for which to compute the center and scale.
        :type data: array-like
        :return: A tuple of center and scale.
        :rtype: tuple
        """
        if self.method == "median":
            center = np.median(data)
            scale = mad(data) / norm.ppf(0.75)
        else:
            center = np.mean(data)
            scale = np.std(data)
        return center, scale

    def fit(self, X, y=None):
        """Fit the scaler to the features X and optionally to the target y.

        :param X: Features to fit.
        :type X: array-like
        :param y: Optional target to fit.
        :type y: array-like, optional
        :return: The fitted scaler.
        :rtype: StandardizingScaler
        """
        self.x_centers, self.x_scales = self._vectorize_data(
            X, self._compute_center_scale
        )
        if y is not None:
            self.y_center, self.y_scale = self._compute_center_scale(np.array(y))
        return self

    def _vectorize_data(self, data, func):
        """Apply a function to each column of the data and return a vector of the results.

        :param data: The data to vectorize.
        :type data: array-like
        :param func: The function to apply to each column of the data.
        :type func: callable
        :return: A vector of the function results.
        :rtype: array-like
        """
        if isinstance(data, pd.DataFrame):
            return np.array([func(data.iloc[:, i]) for i in range(data.shape[1])]).T
        elif len(data.shape) == 1:
            return func(data)
        return np.array([func(data[:, i]) for i in range(data.shape[1])]).T

    def _apply_transform(self, data, centers, scales):
        if len(data.shape) == 1:
            return (data - centers) / scales
        return (data - centers.reshape(1, -1)) / scales.reshape(1, -1)

    def transform(self, X, y=None):
        """Transform the features X and optionally the target y using the fitted scaler.

        :param X: Features to transform.
        :type X: array-like
        :param y: Optional target to transform.
        :type y: array-like, optional
        :return: The transformed features and optionally the transformed target.
        :rtype: tuple
        """
        X_transformed = self._apply_transform(X, self.x_centers, self.x_scales)
        if y is None:
            return X_transformed
        y_transformed = (np.array(y) - self.y_center) / self.y_scale
        return X_transformed, self._to_dataframe(y, y_transformed)

    def inverse_transform(self, X=None, y=None):
        """Apply the inverse transformation to the features X and optionally the target y.

        :param X: Transformed features to inverse transform.
        :type X: array-like, optional
        :param y: Transformed target to inverse transform.
        :type y: array-like, optional
        :return: The original features and target.
        :rtype: tuple
        """
        X_inverted, y_inverted = None, None
        if X is not None:
            X_inverted = self._apply_inverse_transform(X, self.x_centers, self.x_scales)
        if y is not None:
            y_inverted = (np.array(y) * self.y_scale) + self.y_center

        return (self._to_dataframe(X, X_inverted), self._to_dataframe(y, y_inverted))
