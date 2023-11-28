import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t

from remodels.transformers.BaseScaler import BaseScaler


class PITScaler(BaseScaler):
    def __init__(self, distribution="normal", nu=8):
        self.distribution = distribution
        self.nu = nu
        self.empirical_cdfs = {}
        self.y_column_name = None

    def fit(self, X, y=None):
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

    def transform(self, X, y=None):
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

    def inverse_transform(self, X=None, y=None):
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
