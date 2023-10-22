from sklearn.base import BaseEstimator, TransformerMixin

class BaseScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)