"""Evaluation metrics, forecast plots, and failure analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # avoid divide-by-zero for near-zero overnight traffic
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    return {"MAE": mae, "MAPE": mape, "RMSE": rmse}


def plot_forecast_vs_actual(actual: pd.Series, predicted: np.ndarray, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual.index, actual.values, label="Actual", linewidth=1.0)
    ax.plot(actual.index, predicted, label="Predicted", linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("Internet traffic")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(results_df: pd.DataFrame, metric: str, out_path: str):
    """results_df: rows=area, columns=model, values=metric."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    results_df.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Comparison Across Models and Areas")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_failure_analysis(actual: pd.Series, predicted: np.ndarray, out_path: str, title: str, top_k=20):
    errors = np.abs(actual.values - predicted)
    worst_idx = np.argsort(errors)[-top_k:]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual.index, actual.values, label="Actual", linewidth=0.8, color="#3b6fa0")
    ax.plot(actual.index, predicted, label="Predicted", linewidth=0.8, color="#e07b39", linestyle="--")
    ax.scatter(actual.index[worst_idx], actual.values[worst_idx], color="red", s=20,
               zorder=5, label=f"Top-{top_k} worst errors")
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
