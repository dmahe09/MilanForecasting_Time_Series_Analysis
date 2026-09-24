"""
Exploratory analysis for the Milan internet traffic dataset.
Produces all figures required by Task 2 of the assignment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose


def top_n_areas(wide_df: pd.DataFrame, n: int = 3):
    totals = wide_df.sum(axis=0).sort_values(ascending=False)
    return totals.index[:n].tolist(), totals


def plot_total_traffic_distribution(totals: pd.Series, out_path: str):
    """Fig 1: distribution of total traffic across all areas (log scale)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    log_totals = np.log1p(totals.values)
    ax.hist(log_totals, bins=60, color="#3b6fa0", edgecolor="white")
    ax.set_xlabel("log(1 + total internet traffic)")
    ax.set_ylabel("Number of areas")
    ax.set_title("Distribution of Total Internet Traffic Across 10,000 Areas")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_first_two_weeks(wide_df: pd.DataFrame, area_ids: list, labels: list, out_path: str):
    """Fig 2: first-two-weeks time series for the 5 required areas."""
    start = wide_df.index.min()
    end = start + pd.Timedelta(days=14)
    window = wide_df.loc[start:end]

    fig, ax = plt.subplots(figsize=(11, 5))
    for aid, label in zip(area_ids, labels):
        if aid in window.columns:
            ax.plot(window.index, window[aid], label=label, linewidth=1.0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Internet traffic (activity units)")
    ax.set_title("Internet Traffic — First Two Weeks")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_adf_test(series: pd.Series) -> dict:
    series = series.dropna()
    stat, pvalue, *_ = adfuller(series, autolag="AIC")
    return {"adf_statistic": stat, "p_value": pvalue, "stationary": pvalue < 0.05}


def plot_decomposition(series: pd.Series, period: int, out_path: str, title: str):
    """Additive seasonal decomposition (daily period = 144 steps @ 10 min)."""
    clean = series.dropna()
    result = seasonal_decompose(clean, model="additive", period=period, extrapolate_trend="freq")
    fig = result.plot()
    fig.set_size_inches(9, 7)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_acf_pacf(series: pd.Series, lags: int, out_path: str, title: str):
    clean = series.dropna()
    acf_vals = acf(clean, nlags=lags)
    pacf_vals = pacf(clean, nlags=lags)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].stem(range(len(acf_vals)), acf_vals)
    axes[0].set_title(f"ACF — {title}")
    axes[1].stem(range(len(pacf_vals)), pacf_vals)
    axes[1].set_title(f"PACF — {title}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def detect_anomalies_3sigma(series: pd.Series):
    """3-sigma rule anomaly flagging. Returns boolean mask."""
    mu, sigma = series.mean(), series.std()
    return series > (mu + 3 * sigma)


def plot_anomalies(series: pd.Series, out_path: str, title: str):
    mask = detect_anomalies_3sigma(series)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values, linewidth=0.7, color="#3b6fa0")
    ax.scatter(series.index[mask], series.values[mask], color="red", s=12, zorder=5, label="Anomaly (>3σ)")
    ax.set_title(f"Anomaly Detection — {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
