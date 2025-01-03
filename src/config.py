from enum import StrEnum

import pandas as pd
from pyprojroot.here import (
    here,
)  # This package is useful to track the root directory of the package

from src.period import Period

logger_name = "energy-prediction"

# Useful locations of files
root_folder = here()

path_data = root_folder / "data"
path_data_raw = path_data / "raw"
path_data_interim = path_data / "interim"
path_data_final = path_data / "final"
path_data_load = path_data_interim / "energy_load.csv"
path_log = root_folder / "forecasting.log"
path_models = root_folder / "models"
path_preprocessors = root_folder / "preprocessors"

path_html = root_folder / "output"
path_template = path_html / "template.html"
path_output_html = root_folder / "index.html"

random_state = 314159  # Random state for reproducibility
cutoff_mape = 0.1
FORMAT_DATE_CSV = "%d.%m.%Y %H:%M"

# Training and testing parameters
predict_size = 24
refit_size = 7
delta_forecast = pd.Timedelta(hours=predict_size)
train_size = pd.Timedelta(days=3 * 365)
number_folds = 10
delta_val = pd.Timedelta(hours=predict_size * refit_size * number_folds)
start_train = "2019-01-01 00:00+00:00"
end_train_default = "2024-06-25 00:00+00:00"

# Periods for cyclical features (in hours)
max_periods_manual = [12, 24, 24 * 3, 24 * 7, 24 * 30 * 6, 24 * 365]
periods = [
    Period(name="daily", n_periods=12, column="hour", input_range=(1, 24)),
    Period(name="weekly", n_periods=7, column="dayofweek", input_range=(0, 6)),
    Period(name="monthly", n_periods=12, column="month", input_range=(1, 12)),
    Period(name="quarterly", n_periods=4, column="quarter", input_range=(1, 4)),
    Period(name="yearly", n_periods=12, column="dayofyear", input_range=(1, 365)),
]
# Lag features to consider
lags_consider = list(range(1, 24))
n_hyperparameters_trials = 20


# API configs

api_base_url = "https://web-api.tp.entsoe.eu/api"
FORMAT_DATE_API = "%Y%m%d%H00"
API_COUNTRY_CODE = "FR"


class APICountry(StrEnum):
    FR: str = "10YFR-RTE------C"


class APIProcessType(StrEnum):
    ACTUAL: str = "A16"
    DAY_AHEAD_FORECAST: str = "A01"


class APIDocumentType(StrEnum):
    SYSTEM_TOTAL_LOAD: str = "A65"


COL_ACTUAL = "Actual Total Load [MW] - France (FR)"
COL_FORECAST = "Day-ahead Total Load Forecast [MW] - France (FR)"
INDEX_NAME = "Time (UTC)"
