from dataclasses import dataclass


@dataclass
class Period:
    """Class abstraction for the information required to encode a period using RBF."""

    name: str
    n_periods: int
    column: str
    input_range: tuple[int, int]
