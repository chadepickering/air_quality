"""
Step 4 batch pipeline: raw_readings → processed_features.

Entry point: build_processed_features(con) or python -m streaming.feature_engineering

Train/val/test split constants are defined here and must be imported by all
downstream model training scripts to prevent split boundary drift.
"""
from __future__ import annotations

from datetime import date

import duckdb
import numpy as np
import pandas as pd

from ingestion.database import DB_PATH, write_processed_features
from ingestion.station_registry import load_neighbor_index, load_stations
from streaming.sensor_validation import apply_validation
from streaming.spatial_weights import compute_spatial_features

# ---------------------------------------------------------------------------
# Split constants — import these in all model training scripts; do not re-derive.
# ---------------------------------------------------------------------------
TRAIN_END = date(2025, 9, 30)   # inclusive — ~4.5 years of training data
VAL_END   = date(2025, 12, 31)  # Oct–Dec 2025: hyperparameter tuning and λ grid search
# Test set: Jan 2026 onward (AQS ~2-month publication lag makes this the effective ceiling)

PARAMETERS = ["pm25", "no2", "o3", "pm10", "co"]

SCHEMA_COLS = [
    "station_id", "timestamp",
    "pm25", "no2", "o3", "pm10", "co",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "pm25_roll3", "pm25_roll6", "pm25_roll24",
    "pm25_lag1", "pm25_lag3", "pm25_lag24",
    "spatial_pm25_lag1", "spatial_pm25_lag3", "spatial_pm25_roll6",
    "spatial_no2_lag1", "spatial_o3_lag1", "spatial_elev_diff",
    "split",
]


def assign_split(ts: pd.Timestamp) -> str:
    """Return 'train', 'val', or 'test' for a UTC timestamp."""
    d = ts.date()
    if d <= TRAIN_END:
        return "train"
    if d <= VAL_END:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def _impute_series(s: pd.Series) -> pd.Series:
    """
    Tiered imputation for a single hourly parameter series (DatetimeIndex).

    Tiers:
        1–3 hr gaps  → linear interpolation between flanking valid readings.
        4–24 hr gaps → same-hour-of-day median from the prior 7 days.
        >24 hr gaps  → left as NaN (extended outage; flag propagated downstream).

    The 1–3 hr interpolation uses future data (the post-gap anchor), which is
    acceptable for batch training data. Imputation fill values use only past data.
    """
    if not s.isna().any():
        return s

    s = s.copy()
    is_nan = s.isna()
    run_id = (is_nan != is_nan.shift()).cumsum()

    for rid, block in s.groupby(run_id):
        if not block.isna().all():
            continue
        gap_len = len(block)

        if 1 <= gap_len <= 3:
            start_ts  = block.index[0]
            end_ts    = block.index[-1]
            before_ts = start_ts - pd.Timedelta(hours=1)
            after_ts  = end_ts   + pd.Timedelta(hours=1)
            if before_ts in s.index and after_ts in s.index:
                v0, v1 = s[before_ts], s[after_ts]
                if pd.notna(v0) and pd.notna(v1):
                    for i, ts in enumerate(block.index):
                        s[ts] = v0 + (v1 - v0) * (i + 1) / (gap_len + 1)

        elif 4 <= gap_len <= 24:
            for ts in block.index:
                window_start = ts - pd.Timedelta(days=7)
                window = s[window_start : ts - pd.Timedelta(hours=1)]
                same_hr = window[window.index.hour == ts.hour].dropna()
                if not same_hr.empty:
                    s[ts] = same_hr.median()

        # gap_len > 24: leave as NaN

    return s


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df = df.copy()
    df["hour_of_day"] = idx.hour
    df["day_of_week"] = idx.dayofweek   # 0=Monday, 6=Sunday
    df["month"]       = idx.month
    df["is_weekend"]  = idx.dayofweek >= 5
    return df


def _add_rolling_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling windows and lag features for PM2.5.

    Rolling windows use min_periods=1 so the first few rows get partial-window
    averages rather than NaN.  Lag features at the start of the series are NaN
    by construction — this is correct; models learn to handle short context.
    """
    df = df.copy()
    if "pm25" not in df.columns:
        for col in ["pm25_roll3", "pm25_roll6", "pm25_roll24",
                    "pm25_lag1", "pm25_lag3", "pm25_lag24"]:
            df[col] = np.nan
        return df
    pm25 = df["pm25"]
    df["pm25_roll3"]  = pm25.rolling(3,  min_periods=1).mean()
    df["pm25_roll6"]  = pm25.rolling(6,  min_periods=1).mean()
    df["pm25_roll24"] = pm25.rolling(24, min_periods=1).mean()
    df["pm25_lag1"]   = pm25.shift(1)
    df["pm25_lag3"]   = pm25.shift(3)
    df["pm25_lag24"]  = pm25.shift(24)
    return df


# ---------------------------------------------------------------------------
# Main batch pipeline
# ---------------------------------------------------------------------------

def build_processed_features(
    con: duckdb.DuckDBPyConnection,
    neighbor_index: dict | None = None,
    stations: list[dict] | None = None,
) -> int:
    """
    Full batch pipeline: raw_readings → processed_features.

    Steps:
        1. Load raw_readings; apply range-based sensor validation.
        2. Pivot to wide format (station × timestamp), set invalid readings to NaN.
        3. Reindex each station to a full hourly time range; impute gaps.
        4. Add temporal features (hour, day_of_week, month, is_weekend).
        5. Add PM2.5 rolling windows (3/6/24 hr) and lags (1/3/24 hr).
        6. Compute six spatial features via Epanechnikov kernel (pass 2 —
           requires all stations' imputed data to be available first).
        7. Assign train/val/test split labels.
        8. Write to processed_features table (ON CONFLICT DO NOTHING).

    Returns total rows written.

    Leakage notes:
        - Rolling stats and lag features use only past data by construction.
        - Spatial features also look back (lag/roll) using only past neighbor data.
        - 7-day lookback medians for imputation are strictly causal.
        - Scalers (z-score normalization) are NOT applied here; they must be
          fit on train rows only in the model training scripts.
    """
    if neighbor_index is None:
        neighbor_index = load_neighbor_index()
    if stations is None:
        stations = load_stations()

    elevation_lookup: dict[str, float] = {
        str(s["station_id"]): float(s["elevation_m"]) if s.get("elevation_m") is not None else 0.0
        for s in stations
    }

    # --- Step 1: load and validate ---
    print("Loading raw_readings from DuckDB...")
    df_raw = con.execute("""
        SELECT station_id, parameter, value, unit, timestamp, quality_flag
        FROM raw_readings
        ORDER BY station_id, timestamp, parameter
    """).df()
    print(f"  {len(df_raw):,} rows loaded.")

    print("Applying sensor validation...")
    df_raw = apply_validation(df_raw)
    n_invalid = int((df_raw["quality_flag"] >= 2).sum())
    print(f"  {n_invalid:,} readings flagged invalid by range check.")

    # Treat quality_flag >= 2 as missing before pivoting
    df_raw.loc[df_raw["quality_flag"] >= 2, "value"] = np.nan

    # --- Step 2: pivot to wide format ---
    print("Pivoting to wide format...")
    # Unstack is faster than pivot_table for unique-index data
    df_indexed = (
        df_raw[["station_id", "parameter", "value", "timestamp"]]
        .set_index(["station_id", "timestamp", "parameter"])["value"]
        .unstack("parameter")
    )
    df_indexed.columns.name = None
    df_wide = df_indexed.reset_index()
    print(f"  Wide format: {len(df_wide):,} station-hour rows.")

    # --- Step 3 / 4 / 5: per-station imputation + features (pass 1) ---
    print("Processing stations (imputation + temporal + rolling/lag features)...")
    station_dfs: dict[str, pd.DataFrame] = {}

    for sid, sdf in df_wide.groupby("station_id"):
        sdf = sdf.set_index("timestamp").sort_index().drop(columns=["station_id"])

        # Ensure all parameter columns exist
        for param in PARAMETERS:
            if param not in sdf.columns:
                sdf[param] = np.nan

        # Reindex to full hourly time range (preserves index name "timestamp")
        full_idx = pd.date_range(
            start=sdf.index.min(),
            end=sdf.index.max(),
            freq="h",
            name="timestamp",
        )
        sdf = sdf.reindex(full_idx)

        # Impute each parameter independently
        for param in PARAMETERS:
            sdf[param] = _impute_series(sdf[param])

        sdf = _add_temporal_features(sdf)
        sdf = _add_rolling_lag_features(sdf)

        station_dfs[str(sid)] = sdf
        print(f"  {sid}: {len(sdf):,} rows")

    # --- Step 6: spatial features (pass 2 — all stations' data now available) ---
    print("Computing spatial features...")
    for sid, sdf in station_dfs.items():
        neighbors = neighbor_index.get(sid, [])
        station_dfs[sid] = compute_spatial_features(
            sid, sdf, neighbors, station_dfs, elevation_lookup
        )

    # --- Step 7 / 8: assemble + write ---
    print("Assembling final DataFrame...")
    frames = []
    for sid, sdf in station_dfs.items():
        sdf = sdf.copy()
        sdf["station_id"] = sid
        sdf["split"] = pd.Index(sdf.index).map(assign_split)
        sdf = sdf.reset_index()   # "timestamp" index → column
        frames.append(sdf)

    df_out = pd.concat(frames, ignore_index=True)

    # Fill any missing schema columns with NaN and reorder
    for col in SCHEMA_COLS:
        if col not in df_out.columns:
            df_out[col] = np.nan
    df_out = df_out[SCHEMA_COLS]

    print(f"Writing {len(df_out):,} rows to processed_features...")
    n = write_processed_features(con, df_out)
    print(f"Done. {n:,} rows written.")
    return n


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from ingestion.database import initialize_database

    con = initialize_database()
    n = build_processed_features(con)
    print(f"\nStep 4 complete: {n:,} processed feature rows in processed_features.")
    con.close()
