import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LinearlyInterpolateTS(BaseEstimator, TransformerMixin):
    """Custom transformer to interpolate timeseries."""

    def __init__(self, cols, **args):
        self.cols = cols
        self.args = args

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in self.cols:
            X[f"{col}_interpolated"] = X[col].interpolate(**self.args, axis=0)
        return X
