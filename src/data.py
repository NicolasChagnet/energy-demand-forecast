from src import config, data, preprocessing, features
import pandas as pd
import datetime
import numpy as np
import requests
import xmltodict
from src.logger import logger


def load_energy():
    """Loads the energy dataset

    Returns:
        pandas.DataFrame: DataFrame of interest indexed by time.
    """
    df = pd.read_csv(config.path_data_load, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def treat_df(filename):
    # Import CSV
    df = pd.read_csv(filename)
    # Rename columns
    df = df.rename(
        columns={
            "Time (UTC)": "time",
            "Day-ahead Total Load Forecast [MW] - France (FR)": "load_forecast_day_ahead",
            "Actual Total Load [MW] - France (FR)": "load_actual",
        }
    )
    df["time"] = df["time"].apply(lambda x: x.split("-")[0].strip())
    df["time"] = pd.to_datetime(df["time"], utc=True, format=config.format_date)
    # Set the time as index and sample it hourly
    df = df.set_index("time")
    df.asfreq("h")

    for col in df.columns:
        df[col] = df[col].apply(lambda x: np.nan if x == "-" else x)

    return df


def merge_raw_files():
    """Merges all raw files in the folder `/data/raw` into one file in the `interim` folder."""
    logger.info("Merging raw files...")
    list_dfs = [data.treat_df(f) for f in config.path_data_raw.glob("*.csv")]
    merged_df = pd.concat(list_dfs)
    merged_df = merged_df.sort_index()
    merged_df = merged_df.loc[merged_df.index < pd.Timestamp.today(tz=None).strftime("%Y-%m-%d %H:%M")]
    merged_df.to_csv(config.path_data_load)


def build_final_files():
    """Given an interim data file, builds a final training, exogeneous and reference prediction dataset."""
    logger.info("Building final training and exogeneous files...")
    # Load the data
    energy_df = load_energy().asfreq("h")
    # Interpolate the data
    energy_interpolated_df = preprocessing.LinearlyInterpolateTS(
        cols=["load_actual", "load_forecast_day_ahead"],
        method="linear",  # limit_direction="forward"
    ).fit_transform(energy_df)
    # If there are still NaNs in the data (large gaps not taken into account by the interpolation)
    # We just use a backfilling method
    energy_interpolated_df["load_actual_interpolated"] = (
        energy_interpolated_df["load_actual_interpolated"].astype("float").bfill()
    )
    # Separate the data into different elements:
    # the training vector, the reference day-ahead prediction, the exogeneous predictors
    y = energy_interpolated_df["load_actual_interpolated"].asfreq("h")
    X = features.BuildExogeneousFeatures(
        index=y.index,
        periods=config.max_periods_manual,
        max_fourier_order=3,
        country_holidays="FRA",
        order_trend=2,
    )
    # Write data to file
    y.to_csv(config.path_data_final / "ts.csv")
    X.to_csv(config.path_data_final / "exogeneous.csv")


def load_all_data():
    """Returns the required data for training and predicting a model.

    Returns:
        (pd.DataFrame, pd.Series): Exogeneous features X, timeseries y
    """
    y = pd.read_csv(config.path_data_final / "ts.csv", index_col=0)
    y.index = pd.to_datetime(y.index, utc=True)
    y = y.asfreq("h")
    X = pd.read_csv(config.path_data_final / "exogeneous.csv", index_col=0)
    X.index = pd.to_datetime(X.index, utc=True)
    X = X.asfreq("h")
    return X, y["load_actual_interpolated"]


def download_data(start, end, API_KEY):
    """Given a start and end data as well as an API_KEY, queries the entsoe.eu API and returns the missing data.

    Args:
        start (datetime.Timestamp): Start date
        end (datetime.Timestamp): End date
        API_KEY (str): API key for the entsoe.eu database

    Returns:
        bool: Success of the operation
    """
    logger.info("Download in progres...")
    format_date = "%Y%m%d%H00"
    format_csv = "%d.%m.%Y %H:%M"
    # convert to UTC and then string for actual
    start_str = (start).strftime(format_date)
    end_str = (end + pd.Timedelta(hours=1)).strftime(format_date)
    # Build the query
    france_code = "10YFR-RTE------C"
    query = "https://web-api.tp.entsoe.eu/api?documentType=A65"
    query += f"&securityToken={API_KEY}"
    query += f"&outBiddingZone_Domain={france_code}"
    query += "&processType=A16"
    query += f"&periodStart={start_str}&periodEnd={end_str}"

    # Query the API and convert to dataframe
    response_actual = requests.get(query)
    try:
        content = xmltodict.parse(response_actual.content)
        content_ts = content["GL_MarketDocument"]["TimeSeries"]
        if type(content_ts) is dict:
            period = content_ts["Period"]
        elif type(content_ts) is list and len(content_ts) >= 1:
            period = content_ts[0]["Period"]
        else:
            raise ValueError("REST data not handled...")
        timeinterval_s = pd.to_datetime(period["timeInterval"]["start"], utc=True)
        timeinterval_e = pd.to_datetime(period["timeInterval"]["end"], utc=True) - pd.Timedelta(hours=1)
        start_str = timeinterval_s.strftime(format_date)
        end_str = timeinterval_e.strftime(format_date)
        data_actual = period["Point"]
        data_actual = [x["quantity"] for x in data_actual]

        idx = pd.date_range(timeinterval_s, timeinterval_e, freq="h")
        idx_name = [f"{x.strftime(format_csv)} - {(x+pd.Timedelta(hours=1)).strftime(format_csv)}" for x in idx]

        logger.info(f"Saving data from {start_str} until {end_str} to file!")
        data_actual = pd.DataFrame(
            {
                "Actual Total Load [MW] - France (FR)": (
                    data_actual if data_actual is not None else ["-" for x in idx_name]
                ),
                "Day-ahead Total Load Forecast [MW] - France (FR)": ["-" for x in idx_name],
            },
            index=idx_name,
        )
        data_actual.index.name = "Time (UTC)"
        data_actual.to_csv(config.path_data_raw / f"Total Load - Day Ahead _ Actual_{start_str}-{end_str}.csv")

        return True
    except KeyError:
        logger.warn("Error downloading `Actual Total Load`. Returning null data.")
        return False
