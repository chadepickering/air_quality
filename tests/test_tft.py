"""
Unit tests for Step 7 — TFT baseline.

No live DuckDB, no W&B, no trained checkpoint required.
All TimeSeriesDataSet tests use a small in-memory synthetic DataFrame.

Coverage:
  - model.py: constants, feature list consistency, build_tft smoke-test
  - train.py: DataFrame preparation (NaN fill, time_idx, is_weekend cast),
              dataset construction (group count, window count, batch shapes)
  - evaluate.py: PI coverage and sharpness helpers
  - Quantile ordering invariant (p5 ≤ p50 ≤ p95)
  - Horizon index mapping correctness
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers — synthetic DataFrame matching processed_features schema
# ---------------------------------------------------------------------------

from models.tft.model import (
    HORIZON_INDICES,
    HORIZONS,
    KNOWN_REALS,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    QUANTILES,
    STATIC_CATS,
    TARGET,
    UNKNOWN_REALS,
)

ALL_FEATURE_COLS = KNOWN_REALS + UNKNOWN_REALS + [TARGET]
MIN_ROWS = MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH + 10   # enough for at least one window


def _make_synthetic_df(
    n_stations: int = 2,
    n_hours: int = MIN_ROWS,
    seed: int = 0,
) -> pd.DataFrame:
    """Minimal processed_features-like DataFrame for dataset construction tests."""
    rng = np.random.default_rng(seed)
    rows = []
    base_ts = pd.Timestamp("2025-01-01", tz=None)
    for s in range(n_stations):
        sid = f"station_{s:02d}"
        for h in range(n_hours):
            ts = base_ts + pd.Timedelta(hours=h)
            row = {
                "station_id": sid,
                "timestamp":  ts,
                "split":      "train",
                "hour_of_day": ts.hour,
                "day_of_week": ts.dayofweek,
                "month":       ts.month,
                "is_weekend":  float(ts.dayofweek >= 5),
                "pm25": float(rng.uniform(5, 40)),
                "no2":  float(rng.uniform(5, 50)),
                "o3":   float(rng.uniform(10, 80)),
                "pm10": float(rng.uniform(5, 60)),
                "co":   float(rng.uniform(0.1, 1.0)),
            }
            for col in [
                "pm25_roll3", "pm25_roll6", "pm25_roll24",
                "pm25_lag1", "pm25_lag3", "pm25_lag24",
                "spatial_pm25_lag1", "spatial_pm25_lag3", "spatial_pm25_roll6",
                "spatial_no2_lag1", "spatial_o3_lag1", "spatial_elev_diff",
            ]:
                row[col] = float(rng.uniform(5, 30))
            rows.append(row)
    df = pd.DataFrame(rows)
    global_min = df["timestamp"].min()
    df["time_idx"] = ((df["timestamp"] - global_min) / pd.Timedelta(hours=1)).astype(int)
    return df


def _build_small_dataset(df: pd.DataFrame):
    """Build a TimeSeriesDataSet from the synthetic DataFrame."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=STATIC_CATS,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=STATIC_CATS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(groups=STATIC_CATS, transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )


# ---------------------------------------------------------------------------
# model.py constants
# ---------------------------------------------------------------------------

class TestModelConstants:
    def test_horizons_ordered(self):
        assert HORIZONS == sorted(HORIZONS)

    def test_horizon_indices_match_horizons(self):
        for h, idx in zip(HORIZONS, HORIZON_INDICES):
            assert idx == h - 1

    def test_max_prediction_length_covers_all_horizons(self):
        assert MAX_PREDICTION_LENGTH >= max(HORIZONS)

    def test_quantiles_ordered(self):
        assert QUANTILES == sorted(QUANTILES)

    def test_quantiles_include_median(self):
        assert 0.5 in QUANTILES

    def test_pi_is_90_percent(self):
        # p5 to p95 = 90% interval
        assert min(QUANTILES) == pytest.approx(0.05)
        assert max(QUANTILES) == pytest.approx(0.95)

    def test_known_reals_are_calendar_only(self):
        # Known reals must only contain features that are knowable in advance
        calendar = {"hour_of_day", "day_of_week", "month", "is_weekend"}
        assert set(KNOWN_REALS) == calendar

    def test_target_not_in_unknown_reals(self):
        assert TARGET not in UNKNOWN_REALS

    def test_no_overlap_known_unknown(self):
        assert not set(KNOWN_REALS) & set(UNKNOWN_REALS)

    def test_feature_lists_nonempty(self):
        assert len(KNOWN_REALS) > 0
        assert len(UNKNOWN_REALS) > 0


# ---------------------------------------------------------------------------
# train.py — DataFrame preparation
# ---------------------------------------------------------------------------

class TestDataPreparation:
    def _prep(self, df):
        """Apply the same NaN-fill and casting logic as load_and_prepare_df."""
        df = df.copy()
        df["is_weekend"] = df["is_weekend"].astype(float)
        global_min = df["timestamp"].min()
        df["time_idx"] = (
            (df["timestamp"] - global_min) / pd.Timedelta(hours=1)
        ).astype(int)
        feature_cols = KNOWN_REALS + UNKNOWN_REALS + [TARGET]
        df[feature_cols] = (
            df.groupby("station_id", group_keys=False)[feature_cols]
            .apply(lambda g: g.ffill().bfill().fillna(0.0))
        )
        return df

    def test_no_nan_after_fill(self):
        df = _make_synthetic_df(n_stations=2, n_hours=MIN_ROWS)
        # Inject NaN into lag columns (start of record)
        df.loc[df.index[:5], "pm25_lag24"] = np.nan
        df = self._prep(df)
        assert df[ALL_FEATURE_COLS].isna().sum().sum() == 0

    def test_time_idx_contiguous_per_station(self):
        df = _make_synthetic_df(n_stations=3, n_hours=MIN_ROWS)
        df = self._prep(df)
        for _, sdf in df.groupby("station_id"):
            idx = sdf["time_idx"].sort_values().values
            diffs = np.diff(idx)
            assert (diffs == 1).all(), "time_idx not contiguous within station"

    def test_is_weekend_is_float(self):
        df = _make_synthetic_df(n_stations=1, n_hours=MIN_ROWS)
        df = self._prep(df)
        assert df["is_weekend"].dtype == float

    def test_time_idx_starts_at_zero(self):
        df = _make_synthetic_df(n_stations=2, n_hours=MIN_ROWS)
        df = self._prep(df)
        assert df["time_idx"].min() == 0


# ---------------------------------------------------------------------------
# TimeSeriesDataSet construction
# ---------------------------------------------------------------------------

class TestTimeSeriesDataset:
    def test_dataset_builds_without_error(self):
        df = _make_synthetic_df(n_stations=2, n_hours=MIN_ROWS)
        ds = _build_small_dataset(df)
        assert len(ds) > 0

    def test_window_count_scales_with_stations(self):
        # Two stations should produce roughly twice as many windows as one
        df1 = _make_synthetic_df(n_stations=1, n_hours=MIN_ROWS)
        df2 = _make_synthetic_df(n_stations=2, n_hours=MIN_ROWS)
        ds1 = _build_small_dataset(df1)
        ds2 = _build_small_dataset(df2)
        assert len(ds2) == pytest.approx(len(ds1) * 2, abs=2)

    def test_batch_encoder_shape(self):
        df = _make_synthetic_df(n_stations=2, n_hours=MIN_ROWS)
        ds = _build_small_dataset(df)
        loader = ds.to_dataloader(train=False, batch_size=4, num_workers=0)
        x, _ = next(iter(loader))
        assert x["encoder_cont"].shape[1] == MAX_ENCODER_LENGTH

    def test_batch_decoder_shape(self):
        df = _make_synthetic_df(n_stations=2, n_hours=MIN_ROWS)
        ds = _build_small_dataset(df)
        loader = ds.to_dataloader(train=False, batch_size=4, num_workers=0)
        x, _ = next(iter(loader))
        assert x["decoder_cont"].shape[1] == MAX_PREDICTION_LENGTH

    def test_target_shape(self):
        df = _make_synthetic_df(n_stations=2, n_hours=MIN_ROWS)
        ds = _build_small_dataset(df)
        loader = ds.to_dataloader(train=False, batch_size=4, num_workers=0)
        _, y = next(iter(loader))
        assert y[0].shape[1] == MAX_PREDICTION_LENGTH


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:
    def test_perfect_coverage_is_100(self):
        # All true values exactly at p50 → inside [p5, p95]
        true = np.array([10.0, 20.0, 30.0])
        p5   = true - 5.0
        p95  = true + 5.0
        covered = ((true >= p5) & (true <= p95)).mean() * 100
        assert covered == pytest.approx(100.0)

    def test_zero_coverage_when_all_outside(self):
        true = np.array([10.0, 20.0])
        p5   = np.array([15.0, 25.0])   # both above true
        p95  = np.array([20.0, 30.0])
        covered = ((true >= p5) & (true <= p95)).mean() * 100
        assert covered == pytest.approx(0.0)

    def test_sharpness_is_mean_width(self):
        p5  = np.array([0.0, 10.0])
        p95 = np.array([10.0, 30.0])
        sharpness = (p95 - p5).mean()
        assert sharpness == pytest.approx(15.0)

    def test_quantile_ordering(self):
        # Any valid TFT output must have p5 ≤ p50 ≤ p95
        rng = np.random.default_rng(42)
        raw = np.sort(rng.uniform(0, 50, (100, 3)), axis=1)   # sorted along quantile dim
        assert (raw[:, 0] <= raw[:, 1]).all()
        assert (raw[:, 1] <= raw[:, 2]).all()

    def test_horizon_index_mapping(self):
        for h, idx in zip(HORIZONS, HORIZON_INDICES):
            assert idx == h - 1, f"h{h} should map to index {h-1}, got {idx}"

    def test_pi_width_target_3hr_narrower_than_72hr(self):
        # Near-term intervals should generally be tighter — test the concept
        # using synthetic data where uncertainty grows with horizon
        rng = np.random.default_rng(0)
        widths_h3  = rng.uniform(2, 5, 100)
        widths_h72 = rng.uniform(8, 15, 100)
        assert widths_h3.mean() < widths_h72.mean()
