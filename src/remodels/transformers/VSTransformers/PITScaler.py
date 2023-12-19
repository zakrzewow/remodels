"""PITScaler."""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t

from remodels.transformers.BaseScaler import BaseScaler


class PITScaler(BaseScaler):
    """Probability Integral Transform (PIT) Scaler applies a transformation to data based on a specified probability distribution.

    This scaler transforms each feature using the cumulative distribution function (CDF) of the specified distribution,
    effectively mapping the empirical CDF of the data to the target distribution. This technique is often used
    in statistical modeling and forecasting to normalize data or make it conform to a certain distribution.

    The scaler also provides an inverse transformation function to revert the data back to
    its original scale.
    """

    def __init__(self, distribution: str = "normal", nu: int = 8) -> None:
        """Initialize the PIT-Scaler.

        :param distribution: distribution, defaults to "normal"
        :type distribution: str, optional
        :param nu: distribution parameter, defaults to 8
        :type nu: int, optional

        """
        self.distribution = distribution
        self.nu = nu
        self.empirical_cdfs = {}
        self.y_column_name = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "PITScaler":
        """Fit the scaler to the data.

        :param X: Input data.
        :type X: pd.DataFrame
        :param y: Optional, target values (None by default).
        :type y: pd.DataFrame, optional
        :return: Returns self.
        :rtype: PITScaler

        """
        if y is not None:
            y_name = y.columns[0]
            self.y_column_name = y_name
            y = pd.DataFrame(y, columns=[y_name])
            X = pd.concat([X, y], axis=1)

        for column in X.columns:
            sorted_data = np.sort(X[column])
            self.empirical_cdfs[column] = (
                sorted_data,
                np.arange(1, len(sorted_data) + 1) / len(sorted_data),
            )
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]:
        """Transforms the data.

        :param X: Input data to transform.
        :type X: pd.DataFrame
        :param y: Optional, target values (None by default).
        :type y: pd.DataFrame, optional
        :return: Transformed data.
        :rtype: pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]

        """
        transformed_data = pd.DataFrame(index=X.index)
        for column in X.columns:
            sorted_data, empirical_cdf = self.empirical_cdfs[column]
            pit_values = np.interp(X[column], sorted_data, empirical_cdf)
            epsilon = 1e-10  # A small tolerance value
            pit_values = np.clip(pit_values, epsilon, 1 - epsilon)
            transformed_data[column] = self._apply_distribution(pit_values)

        if y is not None and self.y_column_name in self.empirical_cdfs:
            sorted_data, empirical_cdf = self.empirical_cdfs[self.y_column_name]
            pit_values = np.interp(y, sorted_data, empirical_cdf)
            epsilon = 1e-10  # A small tolerance value
            pit_values = np.clip(pit_values, epsilon, 1 - epsilon)
            y_transformed = self._apply_distribution(pit_values)
            y_transformed = self._to_dataframe(y, y_transformed)
            return transformed_data, y_transformed

        return transformed_data

    def _apply_distribution(self, pit_values):
        if self.distribution == "normal":
            return norm.ppf(pit_values)
        elif self.distribution == "student-t":
            return t.ppf(pit_values, df=self.nu)
        else:
            raise ValueError("Invalid distribution type. Use 'normal' or 'student-t'.")

    def inverse_transform(
        self, X: pd.DataFrame = None, y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Inverse transform the features and optionally the target.

        :param X: Transformed features to inverse transform.
        :type X: pd.DataFrame
        :param y: Transformed target to inverse transform.
        :type y: pd.DataFrame, optional
        :return: Original features and optionally original target.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]

        """
        if X is not None:
            inverted_data = pd.DataFrame(index=X.index)
            for column in X.columns:
                sorted_data, empirical_cdf = self.empirical_cdfs[column]
                transformed_values = self._apply_inverse_distribution(X[column])
                inverted_data[column] = np.interp(
                    transformed_values, empirical_cdf, sorted_data
                )
        else:
            inverted_data = None

        if y is not None and self.y_column_name in self.empirical_cdfs:
            sorted_data, empirical_cdf = self.empirical_cdfs[self.y_column_name]
            transformed_values = self._apply_inverse_distribution(y)
            y_inverted = np.interp(transformed_values, empirical_cdf, sorted_data)
            y_inverted = self._to_dataframe(y, y_inverted)
        else:
            y_inverted = None
        return inverted_data, y_inverted

    def _apply_inverse_distribution(self, transformed_values):
        if self.distribution == "normal":
            return norm.cdf(transformed_values)
        elif self.distribution == "student-t":
            return t.cdf(transformed_values, df=self.nu)
        else:
            raise ValueError("Invalid distribution type. Use 'normal' or 'student-t'.")
