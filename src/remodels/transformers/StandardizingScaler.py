import numpy as np
from scipy.stats import norm
from remodels.transformers.BaseScaler import BaseScaler
import pandas as pd

def mad(x):
    median = np.median(x)
    dev = np.abs(x - median)
    return np.mean(dev)

class StandardizingScaler(BaseScaler):
    def __init__(self, method='median'):
        if method not in ['median', 'mean']:
            raise ValueError("Method must be 'median' or 'mean'")
        self.method = method
        self.x_centers = None
        self.x_scales = None
        self.y_center = None
        self.y_scale = None
    
    def _compute_center_scale(self, data):
        if self.method == 'median':
            center = np.median(data)
            scale = mad(data) / norm.ppf(0.75)
        else:
            center = np.mean(data)
            scale = np.std(data)
        return center, scale
    
    def fit(self, X, y=None):
        self.x_centers, self.x_scales = self._vectorize_data(X, self._compute_center_scale)
        if y is not None:
            self.y_center, self.y_scale = self._compute_center_scale(np.array(y))
        return self

    def _vectorize_data(self, data, func):
        if isinstance(data, pd.DataFrame):
            return np.array([func(data.iloc[:, i]) for i in range(data.shape[1])]).T
        elif len(data.shape) == 1:
            return func(data)
        return np.array([func(data[:, i]) for i in range(data.shape[1])]).T

    def _apply_transform(self, data, centers, scales):
        if len(data.shape) == 1:
            return (data - centers) / scales
        return (data - centers.reshape(1, -1)) / scales.reshape(1, -1)

    def transform(self, X, y=None):
        X_transformed = self._apply_transform(X, self.x_centers, self.x_scales)
        if y is None:
            return X_transformed
        y_transformed = (y - self.y_center) / self.y_scale
        return X_transformed, y_transformed
        
    def _apply_inverse_transform(self, data, centers, scales):
        if len(data.shape) == 1:
            return (data * scales) + centers
        return (data * scales.reshape(1, -1)) + centers.reshape(1, -1)

    def inverse_transform(self, X=None, y=None):
        if X is not None:
            X_inverted = self._apply_inverse_transform(X, self.x_centers, self.x_scales)
        if y is not None:
            y_inverted = (y * self.y_scale) + self.y_center

        return X_inverted if X is not None else None, y_inverted if y is not None else None