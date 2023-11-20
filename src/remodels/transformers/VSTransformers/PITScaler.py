"""PITScaler."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.stats import t

from remodels.transformers.BaseScaler import BaseScaler


class PITScaler(BaseScaler):
    """Probability Integral Transform (PIT) Scaler using normal or Student's t-distribution."""

    def __init__(self, distribution="normal", nu=8):
        """Initialize the PITScaler with a specified distribution.

        :param distribution: Distribution type for PIT ('normal' or 'student-t').
        :param nu: Degrees of freedom for Student's t-distribution (used if distribution is 'student-t').
        """
        valid_distributions = ["normal", "student-t"]
        if distribution not in valid_distributions:
            raise ValueError(
                f"Invalid distribution type. Use one of {valid_distributions}."
            )

        self.distribution = distribution
        self.nu = nu
        self.empirical_cdfs = {}

    def fit(self, X, y=None):
        """Fit the scaler by calculating the empirical CDFs for the data.

        :param X: Features to fit.
        :param y: Optional target to fit.
        :return: Fitted scaler.
        """
        if y is not None:
            X = pd.concat([X, pd.DataFrame(y)], axis=1)

        for column in X.columns:
            ranks = X[column].rank(method="average")
            empirical_cdf = ranks / (len(X[column]) + 1)
            self.empirical_cdfs[column] = empirical_cdf
        if y is not None:
            ranks = y.rank(method="average")
            empirical_cdf = ranks / (len(y) + 1)
            self.empirical_cdfs["y"] = empirical_cdf
        return self

    def transform(self, X, y=None):
        """Transform the data using the empirical CDFs and the specified distribution.

        :param X: Features to transform.
        :param y: Optional target to transform.
        :return: Transformed features and optionally transformed target.
        """
        transformed_data = self._transform_columns(X)

        if y is not None:
            y_transformed = self._transform_y(y, "y")
            return transformed_data, y_transformed
        return transformed_data

    def _transform_columns(self, X):
        """Transform the columns of X based on the empirical CDFs.

        :param X: Features to transform.
        :return: Transformed features.
        """
        transformed_data = pd.DataFrame()

        for column in X.columns:
            empirical_cdf = self.empirical_cdfs[column]
            transformed_data[column] = self._transform_data(empirical_cdf)

        return transformed_data

    def _transform_y(self, y, ref_column):
        """Transform the target y based on the empirical CDF of the reference column.

        :param y: Target to transform.
        :param ref_column: Reference column from features.
        :return: Transformed target.
        """
        empirical_cdf = self.empirical_cdfs[ref_column][len(y) :]
        return self._transform_data(empirical_cdf)

    def _transform_data(self, empirical_cdf):
        """Apply the PIT transformation based on the specified distribution.

        :param empirical_cdf: Empirical CDF values to transform.
        :return: Transformed values.
        """
        if self.distribution == "normal":
            return norm.ppf(empirical_cdf)
        elif self.distribution == "student-t":
            return t.ppf(empirical_cdf, df=self.nu)
        else:
            raise ValueError("Invalid distribution type. Use 'normal' or 'student-t'.")

    def inverse_transform(self, X, y=None):
        """Inverse transform the data back to the original scale based on the empirical CDFs.

        :param X: Features to inverse transform.
        :param y: Optional target to inverse transform.
        :return: Original features and optionally original target.
        """
        inverted_data = self._inverse_transform_columns(X)

        if y is not None:
            y_inverted = self._inverse_transform_y(y, X.columns[0])
            return inverted_data, y_inverted

        return inverted_data

    def _inverse_transform_columns(self, X):
        """Inverse transform the columns of X back to the original scale.

        :param X: Features to inverse transform.
        :return: Original features.
        """
        inverted_data = pd.DataFrame()

        for column in X.columns:
            empirical_cdf = self.empirical_cdfs[column]
            inverted_pit = self._inverse_transform_data(X[column])
            inv_ecdf = interp1d(
                empirical_cdf, inverted_pit, bounds_error=False, assume_sorted=True
            )
            inverted_data[column] = inv_ecdf(inverted_pit)

        return inverted_data

    def _inverse_transform_y(self, y, ref_column):
        """Inverse transform the target y back to the original scale.

        :param y: Target to inverse transform.
        :param ref_column: Reference column from features.
        :return: Original target.
        """
        sorted_original_data = np.sort(self.empirical_cdfs[ref_column].index[len(y) :])
        sorted_transformed_data = np.sort(y)
        inv_ecdf = interp1d(
            sorted_transformed_data,
            sorted_original_data,
            bounds_error=False,
            assume_sorted=True,
        )
        return inv_ecdf(y)

    def _inverse_transform_data(self, data):
        """Apply the inverse PIT transformation based on the specified distribution.

        :param data: Transformed values to inverse transform.
        :return: Original values.
        """
        if self.distribution == "normal":
            return norm.cdf(data)
        elif self.distribution == "student-t":
            return t.cdf(data, df=self.nu)
        else:
            raise ValueError("Invalid distribution type. Use 'normal' or 'student-t'.")
