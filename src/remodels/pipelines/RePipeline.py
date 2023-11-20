"""RePipeline."""

from sklearn.pipeline import Pipeline


class RePipeline(Pipeline):
    """Custom implementation of the scikit-learn Pipeline class for additional functionality."""

    def _process_step(self, step, Xt, yt=None, **fit_params):
        """Process a single step of the pipeline, fitting it and transforming the data.

        :param step: The pipeline step (transformer or estimator) to process.
        :param Xt: The transformed input data from the previous step.
        :param yt: The target values. It can be None.
        :param fit_params: Additional fitting parameters.
        :return: The transformed feature data, and target data if provided.
        """
        if yt is not None:
            # When target data is provided, fit and transform with targets.
            step.fit(Xt, yt, **fit_params)
            return step.transform(Xt, yt)
        else:
            # When no target data, fit and transform without targets.
            step.fit(Xt, **fit_params)
            return step.transform(Xt), None

    def fit(self, X, y=None, **fit_params):
        """Fit the pipeline with the input and target data.

        :param X: Input data to fit.
        :param y: Target values.
        :param fit_params: Additional fitting parameters.
        :return: The fitted pipeline.
        """
        Xt, yt = X, y
        # Process all steps except the last one.
        for _, step_process in self.steps[:-1]:
            Xt, yt = self._process_step(step_process, Xt, yt, **fit_params)

        # Fit the last step.
        self.steps[-1][1].fit(Xt, yt, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the pipeline and transform the data.

        :param X: Input data to fit.
        :param y: Target values.
        :param fit_params: Additional fitting parameters.
        :return: The transformed feature data, and optionally target data.
        """
        Xt, yt = X, y
        # Process all steps except the last one using fit_transform if available.
        for _, step_process in self.steps[:-1]:
            if hasattr(step_process, "fit_transform"):
                Xt, yt = (
                    step_process.fit_transform(Xt, yt, **fit_params)
                    if yt is not None
                    else (step_process.fit_transform(Xt, **fit_params), None)
                )
            else:
                Xt, yt = self._process_step(step_process, Xt, yt, **fit_params)

        # Fit and transform the last step.
        final_step = self.steps[-1][1]
        if hasattr(final_step, "fit_transform"):
            return (
                final_step.fit_transform(Xt, yt, **fit_params)
                if yt is not None
                else (final_step.fit_transform(Xt, **fit_params), None)
            )
        else:
            final_step.fit(Xt, yt, **fit_params)
            return (
                final_step.transform(Xt, yt)
                if yt is not None
                else final_step.transform(Xt)
            )

    def transform(self, X, y=None):
        """Apply transforms to the data, and the transform method of the final estimator.

        :param X: Input data to transform.
        :param y: Target values.
        :return: Transformed feature data.
        """
        Xt, yt = X, y
        # Transform all steps except the last one.
        for _, step in self.steps[:-1]:
            Xt, yt = (
                step.transform(Xt, yt) if yt is not None else (step.transform(Xt), None)
            )

        # Transform the last step.
        return self.steps[-1][1].transform(Xt, yt)

    def inverse_transform(self, Xt=None, yt=None):
        """Apply inverse transformations in reverse order of the data.

        :param Xt: Transformed feature data to inverse transform.
        :param yt: Transformed target values.
        :return: Original feature data and target values.
        """
        # Apply inverse transformation for all steps in reverse order.
        for _, step in self.steps[::-1]:
            if yt is not None:
                Xt, yt = step.inverse_transform(Xt, yt)
            else:
                Xt = step.inverse_transform(Xt)
        return Xt, yt if yt is not None else Xt
