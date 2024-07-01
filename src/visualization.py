import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller


def plot_distribution(series, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    if series.nunique() < 100:
        sns.histplot(series, ax=ax, bins=30)
    else:
        sns.kdeplot(series, ax=ax)


def plot_distributions(df):
    n_cols = 3
    cols = df.columns
    n_features = len(cols)
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(24, 24 * n_rows / n_cols))
    if n_features <= n_cols:
        axs = axs.reshape((1, n_cols))
    # fig.tight_layout()
    for i in range(n_features):
        pos_x = i // n_cols
        pos_y = i % n_cols
        plot_distribution(df[cols[i]], ax=axs[pos_x, pos_y])


def plot_series(series=pd.Series([]), label=None, ylabel=None, title=None, ax=None):
    """
    Plots a certain time-series with a 'label', 'ylabel', 'title', 'start' and 'end' of the plot.
    """
    sns.set_theme()
    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 12))
    ax.set_xlabel("Time", fontsize=16)

    if label is not None:
        sns.lineplot(series, label=label, ax=ax)
        ax.legend(fontsize=16)
    else:
        sns.lineplot(series, ax=ax)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=16)
    if title is not None:
        ax.set_title(title, fontsize=24)
    ax.grid(True)
    return ax


def plot_periodogram(frequencies, spectrum, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(24, 12))
    ax.step(frequencies, spectrum, color="purple")
    ax.set_xscale("log")
    day_hours = 24
    week_day = 7
    month_day = 30
    year_month = 12
    times = {
        "4 years": 4 * year_month * month_day * day_hours,
        "2 years": 2 * year_month * month_day * day_hours,
        "year": year_month * month_day * day_hours,
        "6 months": month_day * day_hours * 6,
        "3 months": month_day * day_hours * 3,
        "2 months": month_day * day_hours * 2,
        "month": month_day * day_hours,
        "2 weeks": 2 * week_day * day_hours,
        "week": week_day * day_hours,
        "day": day_hours,
        "12 hours": 12,
        "2 hours": 2,
    }
    ax.set_xlim(0.8 / max(times.values()), 1.2 / min(times.values()))
    ax.set_xticks([1.0 / time for time in times.values()])
    ax.set_xticklabels(
        [f"Every {key}" for key in times.keys()],
        rotation=90,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


# Based off of https://www.kaggle.com/code/tanmay111999/avocado-price-forecast-arima-sarima-detailed#Time-Series-Analysis
def test_stationarity(ts, window=12):
    # Determing rolling statistics
    MA = ts.rolling(window=window).mean()
    MSTD = ts.rolling(window=window).std()

    # Plot rolling statistics:
    plt.figure(figsize=(15, 5))
    orig = plt.plot(ts, color="blue", label="Original")
    mean = plt.plot(MA, color="red", label="Rolling Mean")
    std = plt.plot(MSTD, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean & Standard Deviation")
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(ts, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
