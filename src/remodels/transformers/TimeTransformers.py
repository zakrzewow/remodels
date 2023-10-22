from remodels.transformers.BaseScaler import BaseScaler
import pandas as pd

class DSTAdjuster(BaseScaler):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  

    def transform(self, X: pd.DataFrame, y=None):
        X = X.tz_localize(None).resample('H').mean()
        X = X.fillna((X.shift() + X.shift(-1)) / 2)
        X = X.ffill()

        if y is not None:
            y = pd.Series(y, index=X.index).fillna((y.shift() + y.shift(-1)) / 2)
            y = y.ffill()
            return X, y

        return X