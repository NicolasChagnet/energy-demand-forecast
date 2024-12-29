import holidays
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.deterministic import DeterministicProcess, Fourier

from src import config, utils


def BuildExogeneousFeatures(index, periods=[], max_fourier_order=4, order_trend=1, country_holidays=None):
    """Builds the set of exogeneous features from raw data. These are:
        - a polynomial trend,
        - dummy Fourier features with various periods and at various orders,
        - a dummy column matching the holidays of the country

    Args:
        index: A datetime index parametrizing the timeseries.
        periods (list, optional): List of periods to include. Defaults to [].
        max_fourier_order (int, optional): Maximum fourier order. Defaults to 4.
        order_trend (int, optional): _description_. Defaults to 1.
        country_holidays (str, optional): Country code for holidays. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Build Fourier features
    fouriers = [Fourier(period=period, order=max_fourier_order) for period in periods]
    dp = DeterministicProcess(
        index=index,
        constant=True,  # dummy feature for bias (y-intercept)
        order=order_trend,  # trend
        seasonal=False,
        additional_terms=fouriers,  # annual seasonality (fourier)
        drop=True,  # drop terms to avoid collinearity
    )
    X = dp.in_sample()
    # Add French holidays
    if country_holidays is not None:
        holidays_list = holidays.country_holidays(country_holidays)
        X["holidays"] = X.index.map(lambda x: x in holidays_list).astype("int")
    # Add weekend feature
    X["weekend"] = (X.index.dayofweek >= 5).astype("int")
    # Remove "(",")" from column names for LGBM
    X = X.rename(columns=lambda x: x.replace("(", "_").replace(")", "").replace(",", ""))
    return X
