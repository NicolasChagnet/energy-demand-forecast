import logging
from pathlib import Path

import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient

from src import config as c

logger = logging.getLogger(c.logger_name)


def load_full_data() -> pd.DataFrame:
    """Loads the energy dataset

    Returns:
        pandas.DataFrame: DataFrame of interest indexed by time.
    """
    df = pd.read_csv(c.path_data_load)
    df = df.set_index(c.INDEX_NAME)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def treat_df(filename: str | Path) -> pd.DataFrame:
    # Import CSV
    df = pd.read_csv(filename)
    df[c.INDEX_NAME] = pd.to_datetime(df[c.INDEX_NAME], utc=True)
    # Set the time as index and sample it hourly
    df = df.set_index(c.INDEX_NAME)
    df.asfreq("h")

    for col in df.columns:
        df[col] = df[col].apply(lambda x: np.nan if x == "-" else x)

    return df


def merge_raw_files() -> None:
    """Merges all raw files in the folder `/data/raw` into one file in the `interim` folder."""
    logger.info("Merging raw files...")
    list_dfs = [treat_df(f) for f in c.path_data_raw.glob("*.csv")]
    merged_df = pd.concat(list_dfs)
    merged_df = merged_df.sort_index()
    merged_df = merged_df.loc[merged_df.index < pd.Timestamp.utcnow()]
    merged_df.to_csv(c.path_data_load)


def load_timeseries() -> pd.Series:
    """Returns the required data for training and predicting a model.

    Returns:
        pd.Series: timeseries y
    """
    df = pd.read_csv(c.path_data_interim / "energy_load.csv")
    df = df.set_index(c.INDEX_NAME)
    y = df[c.COL_ACTUAL]
    y.index = pd.to_datetime(y.index, utc=True)
    y = y.asfreq("h")
    return y


def load_timeseries_forecast() -> pd.Series:
    """Returns the required data for training and predicting a model.

    Returns:
        pd.Series: timeseries y
    """
    df = pd.read_csv(c.path_data_interim / "energy_load.csv")
    df = df.set_index(c.INDEX_NAME)
    y = df[c.COL_FORECAST]
    y.index = pd.to_datetime(y.index, utc=True)
    y = y.asfreq("h")
    return y


def download_data(api_key: str, start: pd.Timestamp, end: pd.Timestamp):
    """Given a start and end data as well as an API_KEY, queries the entsoe.eu API and returns the missing data.

    Args:
        start (pd.Timestamp): Start date
        end (pd.Timestamp): End date
        api_key (str): API key for the entsoe.eu database

    Returns:
        bool: Success or failure of the operation
    """
    client = EntsoePandasClient(api_key=api_key)
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True)
    df = client.query_load_and_forecast(country_code=c.API_COUNTRY_CODE, start=start, end=end)
    df.columns = [c.COL_ACTUAL, c.COL_FORECAST]
    df.index.name = c.INDEX_NAME
    df.to_csv(
        c.path_data_raw
        / f"Total Load - Day Ahead _ Actual_{start.strftime(c.FORMAT_DATE_API)}-{end.strftime(c.FORMAT_DATE_API)}.csv"
    )
    return True
