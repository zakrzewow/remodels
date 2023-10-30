import numpy as np
from remodels.transformers.BaseScaler import BaseScaler
from scipy.interpolate import interp1d
from scipy.stats import norm, t
import pandas as pd

class BaseClippingScaler(BaseScaler):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y=None):
        return self

    def _clip_data(self, data):
        condition = data.abs() > self.k
        return np.where(condition, self.k * np.sign(data), data)

    def transform(self, X, y=None):
        X_transformed = self._clip_data(X)
        y_transformed = self._clip_data(y) if y is not None else None
        return self._to_dataframe(X, X_transformed), self._to_dataframe(y, y_transformed)

    def inverse_transform(self, X=None, y=None):
        X_inverted = np.clip(X, -3, 3) if X is not None else None
        y_inverted = np.clip(y, -3, 3) if y is not None else None
        return self._to_dataframe(X, X_inverted), self._to_dataframe(y, y_inverted)
    
class LogClippingScaler(BaseScaler):
    def __init__(self, k=3):
        self.k = k
          
    def _transform_data(self, data):
        condition = data.abs() > self.k
        return np.where(condition, np.sign(data) * (np.log(data.abs() - 2) + 3), data)

    def transform(self, X, y=None):
        X_transformed = self._transform_data(X)
        y_transformed = self._transform_data(y) if y is not None else None
        return self._to_dataframe(X, X_transformed), self._to_dataframe(y, y_transformed)

    def inverse_transform(self, X=None, y=None):
        X_inverted = self._inverse_transform_data(X) if X is not None else None
        y_inverted = self._inverse_transform_data(y) if y is not None else None
        return self._to_dataframe(X, X_inverted), self._to_dataframe(y, y_inverted)

    def _inverse_transform_data(self, data):
        condition = data.abs() > self.k
        return np.where(condition, np.sign(data) * (np.exp(data.abs() - 3) + 2), data)

    
class ArcsinhScaler(BaseScaler):

    def _transform_data(self, data):
        return np.arcsinh(data)

    def _inverse_transform_data(self, data):
        return np.sinh(data)

    def transform(self, X, y=None):
        X_transformed = self._transform_data(X)
        y_transformed = self._transform_data(y) if y is not None else None
        return self._to_dataframe(X, X_transformed), self._to_dataframe(y, y_transformed)

    def inverse_transform(self, X=None, y=None):
        X_inverted = self._inverse_transform_data(X) if X is not None else None
        y_inverted = self._inverse_transform_data(y) if y is not None else None
        return self._to_dataframe(X, X_inverted), self._to_dataframe(y, y_inverted)


class MLogScaler(BaseScaler):
    def __init__(self, c=1/3):
        self.c = c

    def _transform_data(self, data):
        return np.sign(data) * (np.log(data.abs() + 1 / self.c) + np.log(self.c))

    def transform(self, X, y=None):
        X_transformed = self._transform_data(X)
        y_transformed = self._transform_data(y) if y is not None else None
        return self._to_dataframe(X, X_transformed), self._to_dataframe(y, y_transformed)

    def _inverse_transform_data(self, data):
        return np.sign(data) * (np.exp(data.abs() - np.log(self.c)) - 1 / self.c)

    def inverse_transform(self, X=None, y=None):
        X_inverted = self._inverse_transform_data(X) if X is not None else None
        y_inverted = self._inverse_transform_data(y) if y is not None else None
        return self._to_dataframe(X, X_inverted), self._to_dataframe(y, y_inverted)

    
class PolyScaler(BaseScaler):
    def __init__(self, lamb=0.125, c=0.05):
        self.lamb = lamb
        self.c = c

    def _transform_data(self, data):
        return np.sign(data) * ((data.abs() + (self.c / self.lamb)**(1 / (self.lamb - 1))).abs()**self.lamb - (self.c / self.lamb)**(self.lamb / (self.lamb - 1)))

    def transform(self, X, y=None):
        X_transformed = self._transform_data(X)
        y_transformed = self._transform_data(y) if y is not None else None
        return self._to_dataframe(X, X_transformed), self._to_dataframe(y, y_transformed)

    def _inverse_transform_data(self, data):
        return np.sign(data) * ((data.abs() + (self.c / self.lamb)**(self.lamb / (self.lamb - 1)))**(1 / self.lamb) - (self.c / self.lamb)**(1 / (self.lamb - 1)))

    def inverse_transform(self, X=None, y=None):
        X_inverted = self._inverse_transform_data(X) if X is not None else None
        y_inverted = self._inverse_transform_data(y) if y is not None else None
        return self._to_dataframe(X, X_inverted), self._to_dataframe(y, y_inverted)
    