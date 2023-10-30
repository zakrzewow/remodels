from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd

class BaseScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    
    def _to_dataframe(self, original, transformed):
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(transformed, index=original.index, columns=original.columns)
        else:
            return transformed  

class MyPipeline(Pipeline):
    def _process_step(self, step, Xt, yt, **fit_params):
        if yt is not None:
            step.fit(Xt, yt, **fit_params)
            return step.transform(Xt, yt)
        else:
            step.fit(Xt, **fit_params)
            return step.transform(Xt), None

    def fit(self, X, y=None, **fit_params):
        Xt, yt = X, y
        for _, step_process in self.steps[:-1]:
            Xt, yt = self._process_step(step_process, Xt, yt, **fit_params)
            
        self.steps[-1][1].fit(Xt, yt, **fit_params)
        return self
    
    def fit_transform(self, X, y=None, **fit_params):
        Xt, yt = X, y
        for _, step_process in self.steps[:-1]:
            if hasattr(step_process, 'fit_transform'):
                if yt is not None:
                    Xt, yt = step_process.fit_transform(Xt, yt, **fit_params)
                else:
                    Xt = step_process.fit_transform(Xt, **fit_params)
            else:
                Xt, yt = self._process_step(step_process, Xt, yt, **fit_params)

        final_step = self.steps[-1][1]
        if hasattr(final_step, 'fit_transform'):
            return final_step.fit_transform(Xt, yt, **fit_params) if yt is not None else final_step.fit_transform(Xt, **fit_params)
        else:
            final_step.fit(Xt, yt, **fit_params)
            return final_step.transform(Xt, yt) if yt is not None else final_step.transform(Xt)

    def transform(self, X, y=None):
        Xt, yt = X, y
        for _, step in self.steps[:-1]:
            Xt, yt = step.transform(Xt, yt) if yt is not None else (step.transform(Xt), None)
            
        return self.steps[-1][1].transform(Xt, yt)

    def inverse_transform(self, Xt=None, yt=None):
        for _, step in self.steps[::-1]:
            Xt, yt = step.inverse_transform(X=Xt, y=yt)
        return Xt, yt 
    