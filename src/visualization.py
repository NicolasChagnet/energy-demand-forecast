import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller


def plot_distribution(series: pd.Series, ax: plt.Axes | None = None) -> None:
    if ax is None:
        fig, ax = plt.subplots()
    if series.nunique() < 100:
        series.plot.hist(ax=ax, bins=30)
    else:
        series.plot.kde(ax=ax)


def plot_distributions(df: pd.DataFrame) -> None:
    n_cols = 2
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


def plot_series(
    series: pd.Series | None = None,
    label: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plots a certain time-series with a 'label', 'ylabel', 'title', 'start' and 'end' of the plot.
    """
    series = pd.Series([]) if series is None else series
    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 12))
    ax.set_xlabel("Time", fontsize=16)

    if label is not None:
        series.plot.line(label=label, ax=ax)
        ax.legend(fontsize=16)
    else:
        series.plot.line(ax=ax)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=16)
    if title is not None:
        ax.set_title(title, fontsize=24)
    ax.grid(True)
    return ax


def plot_periodogram(frequencies: list[int], spectrum: list[float], ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(24, 12))
    ax.step(frequencies, spectrum, lw=2)
    ax.set_xscale("log")
    day_hours = 24
    week_day = 7
    month_day = 30
    year_month = 12

    font = {
        "weight": "normal",
        "size": 16,
    }

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
    ax.set_xticklabels([f"Every {key}" for key in times.keys()], rotation=90)
    ax.tick_params(axis="both", labelsize=13)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance", fontdict=font)
    ax.set_xlabel("Frequencies", fontdict=font)
    ax.set_title("Periodogram", fontdict=dict(fontsize=20))
    return ax


# Based off of https://www.kaggle.com/code/tanmay111999/avocado-price-forecast-arima-sarima-detailed#Time-Series-Analysis
def test_stationarity(ts: pd.Series, window: int = 12) -> None:
    # Determing rolling statistics
    # MA = ts.rolling(window=window).mean()
    # MSTD = ts.rolling(window=window).std()

    # Plot rolling statistics:
    plt.figure(figsize=(15, 5))
    # orig = plt.plot(ts, color="blue", label="Original")
    # mean = plt.plot(MA, color="red", label="Rolling Mean")
    # std = plt.plot(MSTD, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean & Standard Deviation")
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(ts, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
    for key, value in dftest[4].items():
        dfoutput[f"Critical Value ({key})"] = value
    print(dfoutput)


def plot_superimposed(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    group: str,
    kind: str = "line",
    ax: plt.Axes | None = None,
):
    if ax is None:
        fig, ax = plt.subplots()

    groups = df[group].unique()
    for group_val in groups:
        df_sub = df.loc[df[group] == group_val]
        if hue not in df.columns:
            ax.plot(df_sub[x], df_sub[y], alpha=0.25, color=hue)
        else:
            hues = df[hue].unique()
            for hue_val in hues:
                df_subsub = df_sub.loc[df_sub[hue] == hue_val]
                ax.plot(df_subsub[x], df_subsub[y], alpha=0.25, label=hue_val)

    font = {
        "weight": "normal",
        "size": 16,
    }
    ax.set_xlabel(x.capitalize(), fontdict=font)
    ax.set_ylabel(y.capitalize(), fontdict=font)
    ax.set_title(f"Data grouped by {group}")
