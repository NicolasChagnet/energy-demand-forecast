from pathlib import Path
from typing import Any

import dill


def save_to_file(object: Any, path_to_file: str | Path) -> None:
    with open(path_to_file, "wb") as file:
        dill.dump(object, file)


def load_from_file(path_to_file: str | Path) -> Any:
    with open(path_to_file, "rb") as file:
        object = dill.load(file)
    return object
