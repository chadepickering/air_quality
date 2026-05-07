import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Two-tier validation bounds calibrated on 5-year SCAQMD AQS data (2021-2026).
#
# SUSPECT (quality_flag=1): physically impossible but within known instrument
#   noise — the value is retained in processed_features, not imputed away.
#   Lower bound is 0.0 for every parameter: negative concentrations cannot
#   exist in ambient air, so any negative is flagged suspect regardless of
#   magnitude.  Zeroing or imputing them would introduce upward bias in
#   averages (EPA QA Handbook Vol. II, Section 2.2.3 advises retention of
#   negative filter-correction artifacts).  The upper bound covers high-but-
#   plausible extreme events (wildfires, dust storms).
#
# INVALID (quality_flag=2): instrument malfunction — value is treated as
#   missing and imputed by feature_engineering.py.  The lower bound is set
#   at ~2× the observed noise floor so legitimate noise negatives are never
#   misclassified as malfunctions.  Observed 5-yr noise floors:
#     PM2.5 min = -7.7 µg/m³  → invalid below -15.0
#     NO2   min = -1.7 ppb    → invalid below -10.0
#     O3    min = -0.004 ppm  → invalid below -0.05
#     PM10  min = -9.0 µg/m³  → invalid below -20.0
#     CO    min = -0.2 ppm    → invalid below -1.0
#
# AQS qualifier flags (quality_flag=1) from ingestion are never downgraded —
# this layer only upgrades flags when range checks are more severe.
# ---------------------------------------------------------------------------

SUSPECT_BOUNDS: dict[str, tuple[float, float]] = {
    "pm25": (  0.0,  500.0),   # µg/m³ — Palisades/Eaton fire peak ~400 µg/m³
    "no2":  (  0.0,  500.0),   # ppb   — SCAQMD 5yr max ~95 ppb
    "o3":   (  0.0,    0.5),   # ppm   — LA basin 5yr max ~0.145 ppm
    "pm10": (  0.0, 3000.0),   # µg/m³ — Coachella dust events can exceed 2000
    "co":   (  0.0,   20.0),   # ppm   — near-road LA 5yr max ~10 ppm
}

INVALID_BOUNDS: dict[str, tuple[float, float]] = {
    "pm25": ( -15.0,  5000.0),
    "no2":  ( -10.0,  5000.0),
    "o3":   ( -0.05,    10.0),
    "pm10": ( -20.0, 50000.0),
    "co":   (  -1.0,   500.0),
}


def validate_reading(value: float, parameter: str) -> int:
    """
    Range-based quality flag for a single sensor reading.

    Returns:
        0 — valid: within expected operating range
        1 — suspect: outside normal range but physically plausible
        2 — invalid: impossible value indicating instrument malfunction

    Parameters not in SUSPECT_BOUNDS / INVALID_BOUNDS return 0 (unknown parameter).
    """
    inv_lo, inv_hi = INVALID_BOUNDS.get(parameter, (-1e9, 1e9))
    if value < inv_lo or value > inv_hi:
        return 2

    sus_lo, sus_hi = SUSPECT_BOUNDS.get(parameter, (-1e9, 1e9))
    if value < sus_lo or value > sus_hi:
        return 1

    return 0


def apply_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized range-based validation for a raw_readings DataFrame.

    Updates quality_flag to max(existing_flag, range_flag) — never downgrades
    an existing AQS qualifier flag. Expects columns: value, parameter, quality_flag.

    Returns a copy with an updated quality_flag column.
    """
    df = df.copy()
    range_flags = np.zeros(len(df), dtype=int)

    for param in SUSPECT_BOUNDS:
        mask = (df["parameter"] == param).values
        if not mask.any():
            continue
        vals = df.loc[mask, "value"].values
        inv_lo, inv_hi = INVALID_BOUNDS[param]
        sus_lo, sus_hi = SUSPECT_BOUNDS[param]
        flags = np.where(
            (vals < inv_lo) | (vals > inv_hi), 2,
            np.where((vals < sus_lo) | (vals > sus_hi), 1, 0),
        )
        range_flags[mask] = flags

    df["quality_flag"] = np.maximum(df["quality_flag"].values, range_flags)
    return df
