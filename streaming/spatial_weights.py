"""
Epanechnikov kernel spatial feature computation.

Neighbor index is built in ingestion/station_registry.py (composite distance,
d_cutoff=40 km, λ=0.0005 km²/m²).  This module consumes that index to produce
the six spatial weighted feature columns used in processed_features.
"""
import numpy as np
import pandas as pd


def _weighted_spatial_avg(
    series_list: list[pd.Series],
    weights: list[float],
) -> pd.Series:
    """
    NaN-aware normalized weighted average across neighbor series.

    At each timestamp, re-normalizes weights over neighbors with non-NaN values.
    Returns NaN where all neighbors are NaN or the total valid weight is zero.
    """
    if not series_list:
        return pd.Series(dtype=float)

    weights_arr = np.array(weights, dtype=float)
    values_df = pd.concat(series_list, axis=1)
    values_df.columns = range(len(series_list))

    valid_mask = values_df.notna()
    weighted_sum = (values_df * weights_arr).sum(axis=1, skipna=True)
    valid_weight_sum = (valid_mask * weights_arr).sum(axis=1)
    return weighted_sum / valid_weight_sum.replace(0.0, np.nan)


def compute_spatial_features(
    station_id: str,
    df: pd.DataFrame,
    neighbors: list[tuple[str, float]],
    all_station_dfs: dict[str, pd.DataFrame],
    elevation_lookup: dict[str, float],
) -> pd.DataFrame:
    """
    Compute six Epanechnikov kernel-weighted spatial feature columns.

    Features appended to df:
        spatial_pm25_lag1  — kernel-weighted neighbor PM2.5 at t-1
        spatial_pm25_lag3  — kernel-weighted neighbor PM2.5 at t-3
        spatial_pm25_roll6 — kernel-weighted neighbor 6-hr PM2.5 rolling mean
        spatial_no2_lag1   — kernel-weighted neighbor NO2 at t-1
        spatial_o3_lag1    — kernel-weighted neighbor O3 at t-1
        spatial_elev_diff  — static kernel-weighted absolute elevation difference

    Isolated stations (empty neighbors list) receive NaN for all six columns.
    At timestamps where all neighbors are simultaneously NaN for a feature,
    that feature is NaN for that timestamp.

    Weights in `neighbors` are already normalized to sum to 1.0 by
    station_registry.build_spatial_neighbor_index().
    """
    df = df.copy()
    spatial_cols = [
        "spatial_pm25_lag1", "spatial_pm25_lag3", "spatial_pm25_roll6",
        "spatial_no2_lag1",  "spatial_o3_lag1",   "spatial_elev_diff",
    ]
    for col in spatial_cols:
        df[col] = np.nan

    if not neighbors:
        return df

    target_elev = elevation_lookup.get(station_id) or 0.0

    # Align all neighbor DataFrames to the target's time index once
    aligned: list[tuple[str, float, pd.DataFrame]] = []
    for nid, weight in neighbors:
        ndf = all_station_dfs.get(nid)
        if ndf is None:
            continue
        aligned.append((nid, weight, ndf.reindex(df.index)))

    if not aligned:
        return df

    nids    = [nid   for nid,   _, _   in aligned]
    weights = [w     for _,     w, _   in aligned]
    ndfs    = [ndf   for _,     _, ndf in aligned]

    def _col_or_nan(param: str, shift: int = 0) -> list[pd.Series]:
        out = []
        for ndf in ndfs:
            if param in ndf.columns:
                s = ndf[param]
                out.append(s.shift(shift) if shift else s)
            else:
                out.append(pd.Series(np.nan, index=df.index))
        return out

    df["spatial_pm25_lag1"] = _weighted_spatial_avg(_col_or_nan("pm25", shift=1), weights)
    df["spatial_pm25_lag3"] = _weighted_spatial_avg(_col_or_nan("pm25", shift=3), weights)

    # Use pre-computed pm25_roll6 if available; fall back to computing it
    roll6_series = []
    for ndf in ndfs:
        if "pm25_roll6" in ndf.columns:
            roll6_series.append(ndf["pm25_roll6"])
        elif "pm25" in ndf.columns:
            roll6_series.append(ndf["pm25"].rolling(6, min_periods=1).mean())
        else:
            roll6_series.append(pd.Series(np.nan, index=df.index))
    df["spatial_pm25_roll6"] = _weighted_spatial_avg(roll6_series, weights)

    df["spatial_no2_lag1"] = _weighted_spatial_avg(_col_or_nan("no2", shift=1), weights)
    df["spatial_o3_lag1"]  = _weighted_spatial_avg(_col_or_nan("o3",  shift=1), weights)

    # Elevation difference is static (time-invariant)
    elev_diff = sum(
        w * abs(target_elev - (elevation_lookup.get(nid) or 0.0))
        for nid, w in zip(nids, weights)
    )
    df["spatial_elev_diff"] = elev_diff

    return df
