from __future__ import annotations

import logging

import holidays
import pandas as pd
from sklego.preprocessing import RepeatingBasisFunction

import src.config as c
from src.period import Period

logger = logging.getLogger(c.logger_name)


class ExogBuilder:
    """Builds the set of exogeneous features from raw data. These are:
        - seasonal features encoded with RepeatingBasisFunction
        - a dummy column matching the holidays of the country

    Args:
        periods (list[Period], optional): List of periods to include. Defaults to an empty list.
        country_code (str, optional): Country code for holidays. Defaults to None.
    """

    def __init__(
        self,
        periods: list[Period] | None = None,
        country_code: str | None = None,
    ):
        self.is_fitted = False
        self.periods = periods or list()
        self.RBFSeasonalTransformers = [
            (
                period.name,
                RepeatingBasisFunction(
                    n_periods=period.n_periods,
                    remainder="drop",
                    column=period.column,
                    input_range=period.input_range,
                ),
            )
            for period in self.periods
        ]
        self.holidays_list = holidays.country_holidays(country_code) if country_code is not None else None

    def _get_time_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Builds the necessary calendar columns out of the datetime index."""
        X["dayofyear"] = X.index.dayofyear
        X["dayofweek"] = X.index.dayofweek
        X["quarter"] = X.index.quarter
        X["month"] = X.index.month
        X["hour"] = X.index.hour
        return X

    def build(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Create exogeneous features for a given period.

        Args:
            start_date (pd.Timestamp): Start of the period to consider.
            end_date (pd.Timestamp): End of the period to consider.

        Returns:
            pd.DataFrame: Dataframe containing the exogeneous features and indexed over the period considered.
        """
        # Build index
        date_range = pd.date_range(start=start_date, end=end_date, freq="h")
        X = pd.DataFrame(data=[], index=date_range)
        X = self._get_time_columns(X)
        # Encode seasonal features
        seasons_encoded = []
        for period_name, RBFSeasonalTransformer in self.RBFSeasonalTransformers:
            season_encoded = RBFSeasonalTransformer.fit_transform(X)
            seasons_encoded += [
                pd.DataFrame(
                    data=season_encoded,
                    index=X.index,
                    columns=[f"{period_name}_{i}" for i in range(season_encoded.shape[1])],
                )
            ]
        X_ = pd.concat(seasons_encoded, axis=1)

        # Add the holidays list
        if self.holidays_list is not None:
            X_["holidays"] = X_.index.isin(self.holidays_list).astype("int")

        return X_


class LinearlyInterpolateTS:
    """Custom transformer to interpolate timeseries."""

    def apply(self, y: pd.Series) -> pd.Series:
        y = y.interpolate(method="linear")
        y = y.astype("float").ffill()
        if y.isnull().any():
            logger.warning("Null values remaining in timeseries!")
        return y
