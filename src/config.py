import pandas as pd
from itertools import combinations
from pathlib import Path
from pyprojroot.here import (
    here,
)  # This package is useful to track the root directory of the package

# Useful locations of files
root_folder = here()

path_data = root_folder / "data"
path_data_raw = path_data / "raw"
path_data_interim = path_data / "interim"
path_data_final = path_data / "final"
path_data_load = path_data_interim / "energy_load.csv"
path_log = root_folder / "forecasting.log"
path_models = root_folder / "models"

path_html = root_folder / "output"
path_template = path_html / "template.html"
path_output_html = root_folder / "index.html"

random_state = 314159  # Random state for reproducibility
cutoff_mape = 0.1
format_date = "%d.%m.%Y %H:%M"

# Training and testing parameters
predict_size = 24
refit_size = 7
delta_forecast = pd.Timedelta(hours=predict_size)
delta_val = pd.Timedelta(hours=predict_size * refit_size * 2)
start_train = "2019-01-01 00:00+00:00"
end_train_default = "2024-06-25 00:00+00:00"

# Periods for cyclical features (in hours)
max_periods_manual = [12, 24, 24 * 3, 24 * 7, 24 * 30 * 6, 24 * 365]
# Lag features to consider
lags_consider = []
lags_combinations = list(range(1, 11))
for r in range(len(lags_combinations)):
    lags_consider.extend([tuple(x) for x in combinations(lags_combinations, r=r)])
n_hyperparameters_trials = 10
