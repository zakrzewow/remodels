"""RePipeline."""

from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline


class RePipeline(Pipeline):
    """Custom implementation of the scikit-learn Pipeline class for additional functionality.

    This class extends the standard scikit-learn Pipeline by adding specialized handling
    of steps that involve both features and target data, as well as inverse transformations.
    """

    def _process_step(
        self, step, Xt: pd.DataFrame, yt: pd.DataFrame = None, **fit_params
    ) -> pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]:
        """Process a single step of the pipeline, fitting it and transforming the data.

        :param step: The pipeline step (transformer or estimator) to process.
        :type step: transformer or estimator
        :param Xt: The transformed input data from the previous step.
        :type Xt: pd.DataFrame
        :param yt: The target values. It can be None.
        :type yt: pd.DataFrame
        :param fit_params: Additional fitting parameters.
        :type fit_params: dict
        :return: The transformed feature data, and target data if provided.
        :rtype: pd.DataFrame or Tuple[pd.Dataframe, pd.DataFrame]
        """
        if yt is not None:
            # When target data is provided, fit and transform with targets.
            step.fit(Xt, yt, **fit_params)
            return step.transform(Xt, yt)
        else:
            # When no target data, fit and transform without targets.
            step.fit(Xt, **fit_params)
            return step.transform(Xt), None

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **fit_params
    ) -> "RePipeline":
        """Fit the pipeline with the input and target data.

        :param X: Input data to fit.
        :type X: pd.DataFrame
        :param y: Target values.
        :type y: pd.DataFrame
        :param fit_params: Additional fitting parameters.
        :return: The fitted pipeline.
        :rtype: RePipeline
        """
        Xt, yt = X, y
        # Process all steps except the last one.
        for _, step_process in self.steps[:-1]:
            Xt, yt = self._process_step(step_process, Xt, yt, **fit_params)

        # Fit the last step.
        self.steps[-1][1].fit(Xt, yt, **fit_params)
        return self

    def fit_transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **fit_params
    ) -> pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit the pipeline and transform the data.

        :param X: Input data to fit.
        :type X: pd.DataFrame
        :param y: Target values.
        :type y: pd.DataFrame
        :param fit_params: Additional fitting parameters.
        :type fit_params: list
        :return: The transformed feature data, and optionally target data.
        :rtype: pd.DataFrame or Tuple[pd.Dataframe, pd.DataFrame]
        """
        Xt, yt = X, y
        for _, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                Xt, yt = (
                    step.fit_transform(Xt, yt, **fit_params)
                    if yt is not None
                    else (step.fit_transform(Xt, **fit_params), None)
                )
            else:
                Xt, yt = self._process_step(step, Xt, yt, **fit_params)

        # Fit and transform the last step.
        # Process the last step
        final_step = self.steps[-1][1]
        if hasattr(final_step, "fit_transform"):
            return (
                final_step.fit_transform(Xt, yt, **fit_params)
                if yt is not None
                else (final_step.fit_transform(Xt, **fit_params), None)
            )
        else:
            final_step.fit(Xt, yt, **fit_params)
            return final_step.transform(Xt), yt

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply transforms to the data, and the transform method of the final estimator.

        :param X: Input data to transform.
        :type X: pd.DataFrame
        :param y: Target values.
        :type y: pd.DataFrame (optional)
        :return: Transformed feature data.
        :rtype: Tuple[pd.Dataframe, pd.DataFrame]
        """
        Xt, yt = X, y
        # Transform all steps except the last one.
        for _, step in self.steps[:-1]:
            Xt, yt = (
                step.transform(Xt, yt) if yt is not None else (step.transform(Xt), None)
            )

        # Transform the last step.
        return self.steps[-1][1].transform(Xt, yt)

    def inverse_transform(
        self, Xt: pd.DataFrame = None, yt: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply inverse transformations in reverse order of the data.

        :param Xt: Transformed feature data to inverse transform.
        :type Xt: pd.DataFrame
        :param yt: Transformed target values.
        :type yt: pd.DataFrame
        :return: Original feature data and target values.
        :rtype: Tuple[pd.Dataframe, pd.DataFrame]
        """
        # Apply inverse transformation for all steps in reverse order.
        for _, step in self.steps[::-1]:
            if yt is not None:
                Xt, yt = step.inverse_transform(Xt, yt)
            else:
                Xt = step.inverse_transform(Xt)
        return Xt, yt if yt is not None else Xt
