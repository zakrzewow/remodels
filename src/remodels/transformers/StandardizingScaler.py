import numpy as np
from scipy.stats import norm
from remodels.transformers.BaseScaler import BaseScaler

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
    
    def _compute_center_scale(self, data, is_y=False):
        if self.method == 'median':
            center = np.median(data)
            scale = mad(data) / norm.ppf(0.75)
        else:
            center = np.mean(data)
            scale = np.std(data)
            
        if is_y:
            self.y_center, self.y_scale = center, scale
        else:
            self.x_centers, self.x_scales = np.array(center), np.array(scale)
    
    def fit(self, X, y=None):
        self._compute_center_scale(X)
        if y is not None:
            self._compute_center_scale(np.array(y), is_y=True)
        return self

    def transform(self, X, y=None):
        X_transformed = (X - self.x_centers) / self.x_scales
        if y is None:
            return X_transformed
        
        y_transformed = (y - self.y_center) / self.y_scale
        return X_transformed, y_transformed
    
    def inverse_transform(self, X, y=None):
        X_inverted = (X * self.x_scales) + self.x_centers
        if y is None:
            return X_inverted
        
        y_inverted = (y * self.y_scale) + self.y_center
        return X_inverted, y_inverted