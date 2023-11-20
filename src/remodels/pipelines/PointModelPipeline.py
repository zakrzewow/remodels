"""Point Model Pipeline."""

import pandas as pd
from sklearn.pipeline import Pipeline


class PointModelPipeline(Pipeline):
    """PointModelPipeline."""

    def _process_step(self, step, Xt, yt, **fit_params):
        if yt is not None:
            step.fit(Xt, yt, **fit_params)
            return step.transform(Xt, yt)
        else:
            step.fit(Xt, **fit_params)
            return step.transform(Xt), None

    def fit(self, X, y=None, **fit_params):
        """fit."""
        Xt, yt = X, y
        for _, step_process in self.steps[:-1]:
            Xt, yt = self._process_step(step_process, Xt, yt, **fit_params)

        self.steps[-1][1].fit(Xt, yt, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """fit_transform."""
        Xt, yt = X, y
        for _, step_process in self.steps[:-1]:
            if hasattr(step_process, "fit_transform"):
                if yt is not None:
                    Xt, yt = step_process.fit_transform(Xt, yt, **fit_params)
                else:
                    Xt = step_process.fit_transform(Xt, **fit_params)
            else:
                Xt, yt = self._process_step(step_process, Xt, yt, **fit_params)

        final_step = self.steps[-1][1]
        if hasattr(final_step, "fit_transform"):
            return (
                final_step.fit_transform(Xt, yt, **fit_params)
                if yt is not None
                else final_step.fit_transform(Xt, **fit_params)
            )
        else:
            final_step.fit(Xt, yt, **fit_params)
            return (
                final_step.transform(Xt, yt)
                if yt is not None
                else final_step.transform(Xt)
            )

    def transform(self, X, y=None):
        """transform."""
        Xt, yt = X, y
        for _, step in self.steps[:-1]:
            Xt, yt = (
                step.transform(Xt, yt) if yt is not None else (step.transform(Xt), None)
            )

        return self.steps[-1][1].transform(Xt, yt)

    def inverse_transform(self, Xt=None, yt=None):
        """inverse_transform."""
        for _, step in self.steps[::-1]:
            Xt, yt = step.inverse_transform(X=Xt, y=yt)
        return Xt, yt
