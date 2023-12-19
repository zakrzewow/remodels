"""BaseScaler."""

from typing import Tuple

import pandas as pd


class BaseScaler:
    """Custom scaler base class following scikit-learn's conventions."""

    def fit(self, X, y=None) -> "BaseScaler":
        """Fit the scaler to the data. Placeholder that does nothing.

        :param X: Input data.
        :type X: array-like
        :param y: Optional, target values (None by default).
        :type y: array-like, optional
        :return: Returns self.
        :rtype: BaseScaler
        """
        # No fitting process required for base scaler.
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]:
        """Transforms the data. Placeholder that should be overridden by subclasses.

        :param X: Input data to transform.
        :type X: array-like
        :param y: Optional, target values (None by default).
        :type y: array-like, optional
        :return: Transformed data.
        :rtype: pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        """
        # BaseScaler does not implement transform method.
        raise NotImplementedError("Transform method not implemented.")

    def _to_dataframe(
        self, original: pd.DataFrame, transformed: pd.DataFrame
    ) -> pd.DataFrame:
        """Converts transformed data back to a DataFrame if the original was a DataFrame.

        :param original: Original input data.
        :type original: array-like
        :param transformed: Transformed data.
        :type transformed: array-like
        :return: Transformed data as a DataFrame if original was a DataFrame, otherwise array-like.
        :rtype: pd.DataFrame
        """
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(
                transformed, index=original.index, columns=original.columns
            )
        else:
            return transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Fit to data, then transform it.

        :param X: Features to fit and transform.
        :type X: np.ndarray
        :param y: Optional target to fit and transform.
        :type y: np.ndarray, optional
        :return: Transformed features and optionally transformed target.
        :rtype: tuple or np.ndarray
        """
        # Call the fit method (even if it does nothing in this case)
        self.fit(X, y)

        # Call the transform method and return its result
        return self.transform(X, y)

    def inverse_transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Inverse transformat the data. Placeholder that should be overridden by subclasses.

        :param X: Input data to transform.
        :type X: array-like
        :param y: Optional, target values (None by default).
        :type y: array-like, optional
        :return: Transformed data.
        :rtype: array-like
        """
        # BaseScaler does not implement transform method.
        raise NotImplementedError("Transform method not implemented.")
