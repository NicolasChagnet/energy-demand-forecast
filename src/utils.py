from scipy.signal import periodogram
import numpy as np
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess, Fourier


def compute_periodogram(ts, max_peaks=5, scaling="spectrum"):
    frequencies, spectrum = periodogram(
        ts,
        window="boxcar",
        scaling=scaling,
    )
    max_spectrum = np.flip(np.argsort(spectrum))[:max_peaks]
    return frequencies, spectrum, 1.0 / frequencies[max_spectrum], spectrum[max_spectrum]


def build_deterministic_process(ts, periods=[], fourier_order=3):
    fouriers = [Fourier(period=period, order=fourier_order) for period in periods]

    dp = DeterministicProcess(
        index=ts,
        constant=True,  # dummy feature for bias (y-intercept)
        order=1,  # trend (order 1 means linear)
        seasonal=False,
        additional_terms=fouriers,  # annual seasonality (fourier)
        drop=True,  # drop terms to avoid collinearity
    )
    return dp
