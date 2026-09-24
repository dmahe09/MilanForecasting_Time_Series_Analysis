"""
Data loading & memory management for the Milan Telecommunications dataset.

Raw schema per file (tab-separated, no header), 8 columns:
    0: square_id        (int)
    1: time_interval     (int, ms since epoch)
    2: country_code       (int)
    3: sms_in            (float)
    4: sms_out           (float)
    5: call_in           (float)
    6: call_out          (float)
    7: internet_traffic  (float)

We only need columns 0, 1, 7 for this assignment (square_id, timestamp,
internet traffic). Rows are per (square, country) pair for a given 10-min
interval, so multiple rows can share the same (square_id, time_interval) —
they must be summed across countries to get total traffic for that cell.
"""

import glob
import os
import numpy as np
import pandas as pd


RAW_DTYPES = {
    0: "int32",       # square_id (max 10,000 -> int16 would even fit, use int32 for safety)
    1: "int64",       # epoch ms, needs int64
    7: "float32",     # internet_traffic (float32 halves memory vs float64)
}
USE_COLS = [0, 1, 7]
COL_NAMES = ["square_id", "timestamp_ms", "internet_traffic"]


def _memory_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def load_raw_naive(filepaths):
    """
    Naive baseline loader: default dtypes, all columns, no aggregation.
    Used only to measure the *before* memory footprint for the report.
    """
    frames = []
    for fp in filepaths:
        df = pd.read_csv(
            fp, sep="\t", header=None,
            names=["square_id", "timestamp_ms", "country_code",
                   "sms_in", "sms_out", "call_in", "call_out", "internet_traffic"],
        )
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    return full, _memory_mb(full)


def load_raw_optimized(filepaths, chunksize=500_000):
    """
    Memory-efficient loader.

    Optimizations applied:
      1. usecols: only read square_id, timestamp_ms, internet_traffic
         (drops sms/call columns and country_code at parse time).
      2. Explicit narrow dtypes (int32 / int64 / float32) instead of
         pandas' default int64/float64.
      3. Chunked reading per file (avoids loading an entire ~300MB file
         into memory as Python objects before conversion).
      4. Immediate aggregation: rows are grouped by (square_id, timestamp_ms)
         and summed *within each chunk* before concatenation, since
         internet_traffic must be summed across country codes for the
         same cell/interval. This keeps the running frame small instead
         of carrying one row per country per interval.
      5. NaNs in internet_traffic (intervals with no measured traffic)
         are treated as 0 activity, consistent with the dataset
         documentation (Barlacchi et al., 2015).

    Returns
    -------
    pd.DataFrame with columns [square_id, timestamp_ms, internet_traffic],
    one row per (square_id, timestamp_ms), sorted by square_id then time.
    """
    partials = []
    for fp in filepaths:
        for chunk in pd.read_csv(
            fp, sep="\t", header=None,
            usecols=USE_COLS, names=COL_NAMES,
            dtype={"square_id": "int32", "timestamp_ms": "int64"},
            chunksize=chunksize,
        ):
            chunk["internet_traffic"] = chunk["internet_traffic"].fillna(0.0).astype("float32")
            agg = (
                chunk.groupby(["square_id", "timestamp_ms"], as_index=False)["internet_traffic"]
                .sum()
            )
            partials.append(agg)

    combined = pd.concat(partials, ignore_index=True)
    # A final aggregation is needed because the same (square_id, timestamp_ms)
    # pair can appear in different chunks/files if a day file is chunked.
    combined = (
        combined.groupby(["square_id", "timestamp_ms"], as_index=False)["internet_traffic"]
        .sum()
    )
    combined["square_id"] = combined["square_id"].astype("int32")
    combined["internet_traffic"] = combined["internet_traffic"].astype("float32")
    combined = combined.sort_values(["square_id", "timestamp_ms"]).reset_index(drop=True)
    return combined, _memory_mb(combined)


def to_wide_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the long (square_id, timestamp_ms, internet_traffic) table into a
    wide DataFrame indexed by datetime, one column per square_id. Useful for
    per-area slicing in EDA/forecasting without repeated filtering.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    wide = df.pivot(index="datetime", columns="square_id", values="internet_traffic")
    wide = wide.sort_index()
    return wide


def check_continuity(wide_df: pd.DataFrame, freq: str = "10min") -> dict:
    """
    Verifies whether the loaded data forms an unbroken sequence at the
    expected sampling frequency. Critical to run BEFORE any modeling step:
    silently modeling across missing days corrupts every seasonal/lag
    feature (e.g. a 144-step 'daily' lag no longer corresponds to 24 real
    hours if a day was skipped).

    Returns a dict with the full expected index, the missing timestamps,
    and a boolean 'is_continuous' flag. Raises no error -- calling code
    should inspect the result and decide whether to reindex/interpolate
    or to stop and fetch the missing files.
    """
    full_index = pd.date_range(wide_df.index.min(), wide_df.index.max(), freq=freq)
    missing = full_index.difference(wide_df.index)
    missing_days = sorted(set(missing.normalize().unique()))
    return {
        "expected_steps": len(full_index),
        "actual_steps": len(wide_df.index),
        "missing_steps": len(missing),
        "missing_days": [str(d.date()) for d in missing_days],
        "is_continuous": len(missing) == 0,
    }


def reindex_continuous(wide_df: pd.DataFrame, freq: str = "10min") -> pd.DataFrame:
    """
    Reindexes onto a full, gapless datetime index at `freq`, inserting NaN
    for any missing interval rather than letting downstream code silently
    treat two non-adjacent days as adjacent rows. Modeling code should then
    either restrict itself to the longest gap-free contiguous block, or
    explicitly interpolate short gaps (and document that choice).
    """
    full_index = pd.date_range(wide_df.index.min(), wide_df.index.max(), freq=freq)
    return wide_df.reindex(full_index)


def longest_contiguous_block(wide_df: pd.DataFrame, freq: str = "10min"):
    """
    Given a (possibly gappy) wide DataFrame, returns the (start, end) of the
    longest run of consecutive, gap-free timestamps. Use this to select a
    safe training window when the raw download has missing days.
    """
    reindexed = reindex_continuous(wide_df, freq=freq)
    present = reindexed.notna().any(axis=1)
    if not present.any():
        raise ValueError("No data present at all.")

    best_start, best_end, best_len = None, None, 0
    cur_start, cur_len = None, 0
    for ts, ok in present.items():
        if ok:
            if cur_start is None:
                cur_start = ts
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start, best_end = cur_start, ts
        else:
            cur_start, cur_len = None, 0
    return best_start, best_end, best_len


def list_daily_files(data_dir: str, pattern: str = "sms-call-internet-mi-*.txt"):
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {data_dir}. "
            "Download the Internet-mi files from the Harvard Dataverse link "
            "in the assignment and place them there."
        )
    return files
