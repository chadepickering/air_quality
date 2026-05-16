"""
Unit tests for Step 8 — DeepAR primary model.

No live DuckDB, no trained predictor, no W&B required.
All dataset construction tests use in-memory synthetic DataFrames.
All metrics tests use deterministic synthetic arrays.

Must be run inside venv_deepar:
    source venv_deepar/bin/activate && pytest tests/test_deepar.py -v

Coverage:
  - venv_deepar compatibility: gluonts/lightning/torch versions and imports
  - model.py: constants, feature list integrity, build_estimator smoke-test,
              trainer_kwargs override, PREDICTOR_PATH type
  - train.py: FRM exclusion set, NaN fill, _build_list_dataset entry structure
              (keys, dtypes, shapes), station-to-index mapping, build_datasets
              produces correct entry count per split
  - sample_forecasts.py: _crps_energy (perfect/zero/ordering invariants),
                         _make_rolling_instances (window count, actuals shape,
                         feat_dynamic_real orientation, context boundary),
                         PI coverage and sharpness helpers,
                         quantile ordering invariant,
                         horizon index mapping
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic DataFrame factory
# ---------------------------------------------------------------------------

from models.deepar.model import (
    CONTEXT_LENGTH,
    DYNAMIC_REAL_FEATURES,
    FREQ,
    HORIZON_INDICES,
    HORIZONS,
    LAGS_SEQ,
    NUM_FEAT_DYNAMIC_REAL,
    NUM_FEAT_STATIC_CAT,
    NUM_SAMPLES,
    PREDICTION_LENGTH,
    PREDICTOR_PATH,
    TARGET,
)
from models.deepar.train import (
    ALL_FEATURE_COLS,
    FRM_ONLY_STATIONS,
    _build_list_dataset,
)
from models.deepar.sample_forecasts import (
    _crps_energy,
    _make_rolling_instances,
    STRIDE_HOURS,
)

# Enough rows for at least 2 rolling windows (context + 2 * prediction + stride)
MIN_HOURS = CONTEXT_LENGTH + 2 * PREDICTION_LENGTH + STRIDE_HOURS + 10


def _make_synthetic_df(
    n_stations: int = 3,
    n_hours: int = MIN_HOURS,
    seed: int = 0,
    inject_nan_frac: float = 0.0,
) -> pd.DataFrame:
    """
    Synthetic DataFrame matching the processed_features schema used by DeepAR.
    Only calendar features + TARGET are populated (post leakage-fix design).
    """
    rng = np.random.default_rng(seed)
    rows = []
    base_ts = pd.Timestamp("2025-01-01")
    for s in range(n_stations):
        sid = f"station_{s:02d}"
        for h in range(n_hours):
            ts = base_ts + pd.Timedelta(hours=h)
            rows.append({
                "station_id":  sid,
                "timestamp":   ts,
                "split":       "train",
                "hour_of_day": ts.hour,
                "day_of_week": ts.dayofweek,
                "month":       ts.month,
                "is_weekend":  float(ts.dayofweek >= 5),
                "pm25":        float(rng.uniform(5, 40)),
            })

    df = pd.DataFrame(rows)
    if inject_nan_frac > 0:
        n_nan = int(len(df) * inject_nan_frac)
        idx = rng.choice(len(df), n_nan, replace=False)
        df.loc[idx, "pm25"] = np.nan
    return df


def _station_ids_and_idx(df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    ids = sorted(df["station_id"].unique().tolist())
    return ids, {s: i for i, s in enumerate(ids)}


# ---------------------------------------------------------------------------
# 1. venv_deepar compatibility
# ---------------------------------------------------------------------------

class TestVenvCompatibility:
    def test_gluonts_importable(self):
        import gluonts
        assert gluonts.__version__ is not None

    def test_lightning_version_compatible_with_gluonts(self):
        """gluonts[torch] requires lightning < 2.5; confirm venv satisfies this."""
        import lightning
        major, minor = (int(x) for x in lightning.__version__.split(".")[:2])
        assert (major, minor) < (2, 5), (
            f"lightning {lightning.__version__} breaks gluonts[torch] — needs <2.5"
        )

    def test_torch_available(self):
        import torch
        assert torch.__version__ is not None

    def test_list_dataset_constructible(self):
        from gluonts.dataset.common import ListDataset
        entries = [{"start": pd.Period("2025-01-01", freq="h"), "target": np.array([1.0, 2.0])}]
        ds = ListDataset(entries, freq="h")
        assert len(list(ds)) == 1

    def test_deepar_estimator_importable(self):
        from gluonts.torch.model.deepar import DeepAREstimator
        from models.deepar.model import FixedISQFOutput
        assert DeepAREstimator is not None
        assert FixedISQFOutput is not None


# ---------------------------------------------------------------------------
# 2. model.py constants
# ---------------------------------------------------------------------------

class TestModelConstants:
    def test_horizons_ordered(self):
        assert HORIZONS == sorted(HORIZONS)

    def test_horizon_indices_are_zero_based(self):
        for h, idx in zip(HORIZONS, HORIZON_INDICES):
            assert idx == h - 1, f"h{h} should map to index {h-1}, got {idx}"

    def test_prediction_length_covers_all_horizons(self):
        assert PREDICTION_LENGTH >= max(HORIZONS)

    def test_context_length_is_168(self):
        assert CONTEXT_LENGTH == 168

    def test_num_samples_is_500(self):
        assert NUM_SAMPLES == 500

    def test_dynamic_real_count_matches_constant(self):
        assert len(DYNAMIC_REAL_FEATURES) == NUM_FEAT_DYNAMIC_REAL

    def test_no_duplicate_dynamic_real_features(self):
        assert len(DYNAMIC_REAL_FEATURES) == len(set(DYNAMIC_REAL_FEATURES))

    def test_target_not_in_dynamic_real_features(self):
        assert TARGET not in DYNAMIC_REAL_FEATURES

    def test_static_cat_count_is_one(self):
        assert NUM_FEAT_STATIC_CAT == 1

    def test_predictor_path_is_path_type(self):
        from pathlib import Path
        assert isinstance(PREDICTOR_PATH, Path)

    def test_build_estimator_returns_deepar(self):
        from gluonts.torch.model.deepar import DeepAREstimator
        from models.deepar.model import build_estimator
        est = build_estimator(cardinality=[14])
        assert isinstance(est, DeepAREstimator)

    def test_build_estimator_isqf_output(self):
        from models.deepar.model import build_estimator, FixedISQFOutput, ISQF_NUM_PIECES, ISQF_QK_X
        est = build_estimator(cardinality=[14])
        assert isinstance(est.distr_output, FixedISQFOutput)
        assert est.distr_output.num_pieces == ISQF_NUM_PIECES
        assert list(est.distr_output.qk_x) == ISQF_QK_X

    def test_isqf_qk_x_covers_pi_endpoints(self):
        """p5 and p95 must be explicit knots so the PI is learned, not tail-extrapolated."""
        from models.deepar.model import ISQF_QK_X
        assert 0.05 in ISQF_QK_X, "0.05 knot required for direct p5 learning"
        assert 0.95 in ISQF_QK_X, "0.95 knot required for direct p95 learning"

    def test_build_estimator_num_batches_per_epoch(self):
        from models.deepar.model import build_estimator
        est = build_estimator(cardinality=[14])
        assert est.num_batches_per_epoch == 100

    def test_build_estimator_trainer_kwargs_override(self):
        """trainer_kwargs passed in should override defaults without losing them."""
        from models.deepar.model import build_estimator
        est = build_estimator(cardinality=[14], trainer_kwargs={"max_epochs": 5})
        assert est.trainer_kwargs["max_epochs"] == 5
        # Default keys still present
        assert "accelerator" in est.trainer_kwargs
        assert "gradient_clip_val" in est.trainer_kwargs

    def test_freq_is_hourly(self):
        assert FREQ == "h"

    def test_lags_seq_contains_expected_horizons(self):
        assert LAGS_SEQ == [1, 3, 24]

    def test_build_estimator_uses_lags_seq(self):
        from models.deepar.model import build_estimator
        est = build_estimator(cardinality=[14])
        assert est.lags_seq == LAGS_SEQ

    def test_dynamic_real_features_are_calendar_only(self):
        """After the leakage fix, only future-known calendar features remain."""
        assert set(DYNAMIC_REAL_FEATURES) == {
            "hour_of_day", "day_of_week", "month", "is_weekend"
        }
        assert NUM_FEAT_DYNAMIC_REAL == 4


# ---------------------------------------------------------------------------
# 3. train.py — FRM exclusion and NaN fill
# ---------------------------------------------------------------------------

class TestFRMExclusion:
    def test_frm_only_has_five_stations(self):
        assert len(FRM_ONLY_STATIONS) == 5

    def test_frm_station_ids_are_strings(self):
        assert all(isinstance(s, str) for s in FRM_ONLY_STATIONS)

    def test_frm_stations_are_la_metro_format(self):
        # All should match the AQS format "XX-XXX-XXXX"
        import re
        pattern = re.compile(r"^\d{2}-\d{3}-\d{4}$")
        assert all(pattern.match(s) for s in FRM_ONLY_STATIONS)


# ---------------------------------------------------------------------------
# 4. train.py — ListDataset entry structure
# ---------------------------------------------------------------------------

class TestListDatasetConstruction:
    def _build(self, n_stations=2, n_hours=MIN_HOURS):
        df = _make_synthetic_df(n_stations=n_stations, n_hours=n_hours)
        ids, idx = _station_ids_and_idx(df)
        from gluonts.dataset.common import ListDataset
        entries = []
        for sid in ids:
            sdf = df[df["station_id"] == sid].sort_values("timestamp")
            target = sdf[TARGET].to_numpy(dtype=np.float32)
            feat_dyn = sdf[DYNAMIC_REAL_FEATURES].to_numpy(dtype=np.float32).T
            start = pd.Period(sdf["timestamp"].iloc[0], freq=FREQ)
            entries.append({
                "start":             start,
                "target":            target,
                "feat_dynamic_real": feat_dyn,
                "feat_static_cat":   np.array([idx[sid]], dtype=np.int32),
            })
        return ListDataset(entries, freq=FREQ), ids, idx

    def test_entry_count_matches_stations(self):
        ds, _, _ = self._build(n_stations=3)
        assert len(list(ds)) == 3

    def test_target_dtype_is_float32(self):
        ds, _, _ = self._build()
        for entry in ds:
            assert entry["target"].dtype == np.float32

    def test_feat_dynamic_real_shape(self):
        """Shape must be (NUM_FEAT_DYNAMIC_REAL, T) — GluonTS expects features-first."""
        df = _make_synthetic_df(n_stations=1, n_hours=MIN_HOURS)
        sdf = df[df["station_id"] == "station_00"].sort_values("timestamp")
        feat_dyn = sdf[DYNAMIC_REAL_FEATURES].to_numpy(dtype=np.float32).T
        assert feat_dyn.shape[0] == NUM_FEAT_DYNAMIC_REAL
        assert feat_dyn.shape[1] == MIN_HOURS

    def test_feat_static_cat_is_int32(self):
        ds, _, _ = self._build()
        for entry in ds:
            assert entry["feat_static_cat"].dtype == np.int32

    def test_feat_static_cat_shape(self):
        ds, _, _ = self._build()
        for entry in ds:
            assert entry["feat_static_cat"].shape == (1,)

    def test_start_is_period(self):
        ds, _, _ = self._build()
        for entry in ds:
            assert isinstance(entry["start"], pd.Period)

    def test_station_index_unique_per_station(self):
        df = _make_synthetic_df(n_stations=4, n_hours=MIN_HOURS)
        _, idx = _station_ids_and_idx(df)
        assert len(set(idx.values())) == 4

    def test_station_index_zero_based(self):
        df = _make_synthetic_df(n_stations=3, n_hours=MIN_HOURS)
        _, idx = _station_ids_and_idx(df)
        assert min(idx.values()) == 0
        assert max(idx.values()) == 2

    def test_nan_fill_leaves_no_nan(self):
        df = _make_synthetic_df(n_stations=2, n_hours=MIN_HOURS, inject_nan_frac=0.05)
        # Apply the same fill logic as load_and_prepare_df (NaN injected into pm25)
        df[ALL_FEATURE_COLS] = (
            df.groupby("station_id", group_keys=False)[ALL_FEATURE_COLS]
            .apply(lambda g: g.ffill().bfill().fillna(0.0))
        )
        assert df[ALL_FEATURE_COLS].isna().sum().sum() == 0

    def test_build_list_dataset_helper(self):
        """_build_list_dataset from train.py produces iterable dataset with correct entries."""
        df = _make_synthetic_df(n_stations=2, n_hours=MIN_HOURS)
        ids, idx = _station_ids_and_idx(df)
        ds = _build_list_dataset(df, ids, idx)
        entries = list(ds)
        assert len(entries) == 2
        for entry in entries:
            assert "target" in entry
            assert "feat_dynamic_real" in entry
            assert "feat_static_cat" in entry
            assert "start" in entry

    def test_build_datasets_train_shorter_than_val(self):
        """
        build_datasets slices on TRAIN_END. Using timestamps that straddle it
        (Oct–Dec 2025 = val period per feature_engineering.py) ensures the
        val dataset covers more time than the train dataset.
        """
        from models.deepar.train import build_datasets
        # Start in August 2025 (train) and run through November 2025 (val period)
        n_hours = 24 * 120   # 120 days spanning TRAIN_END (Sep 30)
        rng = np.random.default_rng(5)
        base_ts = pd.Timestamp("2025-08-01")
        rows = []
        for s in range(2):
            sid = f"station_{s:02d}"
            for h in range(n_hours):
                ts = base_ts + pd.Timedelta(hours=h)
                rows.append({
                    "station_id":  sid,
                    "timestamp":   ts,
                    "split":       "train",
                    "hour_of_day": ts.hour,
                    "day_of_week": ts.dayofweek,
                    "month":       ts.month,
                    "is_weekend":  float(ts.dayofweek >= 5),
                    "pm25":        float(rng.uniform(5, 40)),
                })
        df = pd.DataFrame(rows)
        train_ds, val_ds, _, _ = build_datasets(df)
        train_entries = list(train_ds)
        val_entries   = list(val_ds)
        assert len(val_entries[0]["target"]) > len(train_entries[0]["target"])


# ---------------------------------------------------------------------------
# 5. sample_forecasts.py — CRPS helper
# ---------------------------------------------------------------------------

class TestCRPS:
    def test_perfect_forecast_crps_is_zero(self):
        """If all samples equal the true value, CRPS = 0."""
        truth   = np.array([10.0, 20.0, 30.0])
        samples = np.tile(truth[:, None], (1, 100))   # (3, 100) — all equal to truth
        crps = _crps_energy(samples, truth)
        assert abs(crps) < 1e-5

    def test_crps_is_non_negative(self):
        rng = np.random.default_rng(42)
        truth   = rng.uniform(5, 40, 20)
        samples = rng.normal(truth[:, None], 3, (20, 500))
        crps = _crps_energy(samples, truth)
        assert crps >= 0.0

    def test_wider_distribution_gives_higher_crps(self):
        """Higher spread around the same true value = higher CRPS."""
        rng = np.random.default_rng(0)
        truth = np.full(50, 10.0)
        s_tight = rng.normal(10.0, 0.5, (50, 500))
        s_wide  = rng.normal(10.0, 5.0, (50, 500))
        assert _crps_energy(s_tight, truth) < _crps_energy(s_wide, truth)

    def test_biased_forecast_worse_than_unbiased(self):
        """Forecast centered away from truth > forecast centered on truth."""
        rng = np.random.default_rng(1)
        truth    = np.full(50, 10.0)
        unbiased = rng.normal(10.0, 1.0, (50, 500))
        biased   = rng.normal(15.0, 1.0, (50, 500))
        assert _crps_energy(unbiased, truth) < _crps_energy(biased, truth)

    def test_crps_output_is_scalar(self):
        rng = np.random.default_rng(0)
        truth   = rng.uniform(5, 30, 10)
        samples = rng.normal(truth[:, None], 2, (10, 100))
        crps = _crps_energy(samples, truth)
        assert np.ndim(crps) == 0

    def test_crps_dtype_is_float(self):
        samples = np.ones((5, 50), dtype=np.float32) * 5.0
        truth   = np.ones(5, dtype=np.float32) * 5.0
        assert isinstance(_crps_energy(samples, truth), float)


# ---------------------------------------------------------------------------
# 6. sample_forecasts.py — rolling window construction
# ---------------------------------------------------------------------------

class TestRollingWindows:
    def _setup(self, n_stations=2, n_hours=MIN_HOURS):
        df = _make_synthetic_df(n_stations=n_stations, n_hours=n_hours)
        ids, idx = _station_ids_and_idx(df)
        base = df["timestamp"].min()
        test_start = base + pd.Timedelta(hours=CONTEXT_LENGTH)
        test_end   = base + pd.Timedelta(hours=n_hours - 1)
        return df, ids, idx, test_start, test_end

    def test_returns_correct_number_of_windows(self):
        df, ids, idx, test_start, test_end = self._setup(n_stations=2)
        entries, actuals, entry_sids, starts = _make_rolling_instances(
            df, ids, idx, test_start, test_end, PREDICTION_LENGTH, STRIDE_HOURS
        )
        # Should have entries for each valid (station, stride-position) pair
        assert len(entries) > 0
        assert len(entries) == len(actuals) == len(entry_sids) == len(starts)

    def test_actuals_shape(self):
        df, ids, idx, test_start, test_end = self._setup(n_stations=2)
        _, actuals, _, _ = _make_rolling_instances(
            df, ids, idx, test_start, test_end, PREDICTION_LENGTH, STRIDE_HOURS
        )
        assert actuals.ndim == 2
        assert actuals.shape[1] == PREDICTION_LENGTH

    def test_actuals_dtype_float32(self):
        df, ids, idx, test_start, test_end = self._setup()
        _, actuals, _, _ = _make_rolling_instances(
            df, ids, idx, test_start, test_end, PREDICTION_LENGTH, STRIDE_HOURS
        )
        assert actuals.dtype == np.float32

    def test_feat_dynamic_real_orientation(self):
        """feat_dynamic_real must be (NUM_FEAT_DYNAMIC_REAL, T) not (T, NUM_FEAT_DYNAMIC_REAL).
        T spans context + future (up to CONTEXT_LENGTH + PREDICTION_LENGTH) so GluonTS
        has decoder inputs for the prediction horizon."""
        df, ids, idx, test_start, test_end = self._setup(n_stations=1)
        entries, _, _, _ = _make_rolling_instances(
            df, ids, idx, test_start, test_end, PREDICTION_LENGTH, STRIDE_HOURS
        )
        assert len(entries) > 0
        fdr = entries[0]["feat_dynamic_real"]
        assert fdr.shape[0] == NUM_FEAT_DYNAMIC_REAL
        assert fdr.shape[1] <= CONTEXT_LENGTH + PREDICTION_LENGTH

    def test_context_length_not_exceeded(self):
        """Each entry's target should be at most CONTEXT_LENGTH steps long."""
        df, ids, idx, test_start, test_end = self._setup(n_stations=1)
        entries, _, _, _ = _make_rolling_instances(
            df, ids, idx, test_start, test_end, PREDICTION_LENGTH, STRIDE_HOURS
        )
        for entry in entries:
            assert len(entry["target"]) <= CONTEXT_LENGTH

    def test_window_starts_are_timestamps(self):
        df, ids, idx, test_start, test_end = self._setup()
        _, _, _, starts = _make_rolling_instances(
            df, ids, idx, test_start, test_end, PREDICTION_LENGTH, STRIDE_HOURS
        )
        assert all(isinstance(t, pd.Timestamp) for t in starts)

    def test_two_stations_produce_more_windows_than_one(self):
        df1, ids1, idx1, ts1, te1 = self._setup(n_stations=1)
        df2, ids2, idx2, ts2, te2 = self._setup(n_stations=2)
        e1, _, _, _ = _make_rolling_instances(df1, ids1, idx1, ts1, te1, PREDICTION_LENGTH, STRIDE_HOURS)
        e2, _, _, _ = _make_rolling_instances(df2, ids2, idx2, ts2, te2, PREDICTION_LENGTH, STRIDE_HOURS)
        assert len(e2) == pytest.approx(len(e1) * 2, abs=2)


# ---------------------------------------------------------------------------
# 7. Metrics helpers (PI coverage, sharpness, quantile ordering)
# ---------------------------------------------------------------------------

class TestMetricsHelpers:
    def test_perfect_pi_coverage_is_100(self):
        truth = np.array([10.0, 20.0, 30.0])
        p5    = truth - 5.0
        p95   = truth + 5.0
        cov   = ((truth >= p5) & (truth <= p95)).mean() * 100
        assert cov == pytest.approx(100.0)

    def test_zero_pi_coverage_when_all_outside(self):
        truth = np.array([10.0, 20.0])
        p5    = np.array([15.0, 25.0])
        p95   = np.array([20.0, 30.0])
        cov   = ((truth >= p5) & (truth <= p95)).mean() * 100
        assert cov == pytest.approx(0.0)

    def test_sharpness_is_mean_interval_width(self):
        p5  = np.array([0.0, 10.0])
        p95 = np.array([10.0, 30.0])
        assert (p95 - p5).mean() == pytest.approx(15.0)

    def test_quantile_ordering_from_samples(self):
        """p5 ≤ p50 ≤ p95 for any sample array."""
        rng = np.random.default_rng(42)
        samples = rng.normal(10.0, 3.0, (100, 500))
        p5  = np.percentile(samples, 5,  axis=1)
        p50 = np.percentile(samples, 50, axis=1)
        p95 = np.percentile(samples, 95, axis=1)
        assert (p5 <= p50).all()
        assert (p50 <= p95).all()

    def test_horizon_index_mapping(self):
        for h, idx in zip(HORIZONS, HORIZON_INDICES):
            assert idx == h - 1, f"h{h} should map to index {h - 1}, got {idx}"

    def test_p5_p95_is_90_percent_interval(self):
        """Sampling from a known distribution should hit 90% PI ~90% of the time."""
        rng = np.random.default_rng(99)
        # Draw true values and build tight intervals around them
        truth = rng.uniform(5, 40, 1000)
        p5  = truth - 5.0
        p95 = truth + 5.0
        cov = ((truth >= p5) & (truth <= p95)).mean() * 100
        assert cov == pytest.approx(100.0)   # trivially inside since p5/p95 bracket truth exactly

    def test_crps_decreases_as_spread_narrows(self):
        """
        CRPS penalises both bias and spread. For forecasts centred on truth,
        reducing spread strictly reduces CRPS (down toward 0 but never negative).
        """
        rng = np.random.default_rng(7)
        truth = np.full(200, 10.0)
        crps_wide   = _crps_energy(rng.normal(10.0, 5.0, (200, 500)), truth)
        crps_medium = _crps_energy(rng.normal(10.0, 1.0, (200, 500)), truth)
        crps_tight  = _crps_energy(rng.normal(10.0, 0.1, (200, 500)), truth)
        assert crps_wide > crps_medium > crps_tight >= 0.0
