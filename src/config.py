from pathlib import Path

from pyprojroot.here import (
    here,
)  # This package is useful to track the root directory of the package

random_state = 314159  # Random state for reproducibility

# Useful locations of files

root_folder = here()

path_data = root_folder / "data"
path_data_raw = path_data / "raw"
path_data_interim = path_data / "interim"
path_data_final = path_data / "final"
