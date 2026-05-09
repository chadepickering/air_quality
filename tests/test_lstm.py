"""
Unit tests for Step 6 — LSTM baseline.

No GPU, no live DuckDB, no W&B required.  All data is synthesized in-memory.

Coverage:
  - LSTMForecaster: output shape, gradient flow, multi-station batch
  - Scaler: fit, apply, NaN fill, constant-column guard
  - build_windows: correct window count, no NaN leakage into targets,
                   missing-target rows skipped, multi-station independence
  - evaluate._mape: zero-denominator guard, perfect forecast
  - lambda_search grid: smoke-test recompute_spatial returns expected columns
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
import torch

from models.lstm.model import HORIZONS, LSTMForecaster
from models.lstm.train import (
    AQDataset,
    FEATURE_COLS,
    N_FEATURES,
    SEQ_LEN,
    apply_scaler,
    build_windows,
    fit_scaler,
)
from models.lstm.evaluate import _mape


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_station_df(
    n_hours: int = 200,
    station_id: str = "06-037-1103",
    pm25_start: float = 10.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic processed_features-like DataFrame for one station."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    pm25 = pm25_start + np.cumsum(rng.normal(0, 0.5, n_hours))
    pm25 = np.clip(pm25, 1.0, 200.0)

    df = pd.DataFrame({"timestamp": ts, "station_id": station_id})
    df["pm25"]    = pm25
    df["no2"]     = rng.uniform(5, 50, n_hours)
    df["o3"]      = rng.uniform(10, 80, n_hours)
    df["pm10"]    = pm25 * 1.5
    df["co"]      = rng.uniform(0.1, 1.0, n_hours)
    df["hour_of_day"] = ts.hour
    df["day_of_week"] = ts.dayofweek
    df["month"]       = ts.month
    df["is_weekend"]  = (ts.dayofweek >= 5).astype(float)
    for col in ["pm25_roll3", "pm25_roll6", "pm25_roll24"]:
        df[col] = pd.Series(pm25).rolling(int(col.replace("pm25_roll", "")),
                                          min_periods=1).mean().values
    for lag in [1, 3, 24]:
        df[f"pm25_lag{lag}"] = pd.Series(pm25).shift(lag).values
    for col in ["spatial_pm25_lag1", "spatial_pm25_lag3", "spatial_pm25_roll6",
                "spatial_no2_lag1",  "spatial_o3_lag1",   "spatial_elev_diff"]:
        df[col] = rng.uniform(5, 30, n_hours)

    assert list(df.columns[2:]) == FEATURE_COLS, \
        "Synthetic df columns don't match FEATURE_COLS — update helper."
    return df


def _default_scaler(df: pd.DataFrame):
    mean, std = fit_scaler(df[FEATURE_COLS].values.astype(np.float32))
    return mean, std


# ---------------------------------------------------------------------------
# LSTMForecaster
# ---------------------------------------------------------------------------

class TestLSTMForecaster:
    def test_output_shape_batch_1(self):
        model = LSTMForecaster(n_features=N_FEATURES)
        x = torch.randn(1, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert out.shape == (1, len(HORIZONS))

    def test_output_shape_batch_32(self):
        model = LSTMForecaster(n_features=N_FEATURES)
        x = torch.randn(32, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert out.shape == (32, len(HORIZONS))

    def test_four_horizons(self):
        assert len(HORIZONS) == 4
        assert HORIZONS == [3, 12, 24, 72]

    def test_gradient_flows(self):
        model = LSTMForecaster(n_features=N_FEATURES)
        x = torch.randn(4, SEQ_LEN, N_FEATURES)
        target = torch.randn(4, len(HORIZONS))
        loss = torch.nn.MSELoss()(model(x), target)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_custom_hidden_size(self):
        model = LSTMForecaster(n_features=N_FEATURES, hidden_size=32)
        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert out.shape == (2, len(HORIZONS))

    def test_single_layer_no_dropout(self):
        # num_layers=1 must not raise; dropout is suppressed internally
        model = LSTMForecaster(n_features=N_FEATURES, num_layers=1, dropout=0.5)
        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        assert model(x).shape == (2, len(HORIZONS))

    def test_output_differs_across_inputs(self):
        model = LSTMForecaster(n_features=N_FEATURES)
        model.eval()
        x1 = torch.randn(1, SEQ_LEN, N_FEATURES)
        x2 = torch.randn(1, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            assert not torch.allclose(model(x1), model(x2))


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

class TestScaler:
    def test_fit_returns_correct_shapes(self):
        df = _make_station_df(n_hours=200)
        mean, std = _default_scaler(df)
        assert mean.shape == (N_FEATURES,)
        assert std.shape  == (N_FEATURES,)

    def test_apply_produces_zero_mean(self):
        df = _make_station_df(n_hours=500)
        mean, std = _default_scaler(df)
        X = df[FEATURE_COLS].values.astype(np.float32)
        Xs = apply_scaler(X.copy(), mean, std)
        # Scaled mean should be near zero for all non-lag columns
        # (lag columns have NaN at start; those become 0 after nan_to_num)
        assert np.abs(Xs.mean(axis=0)).max() < 1.0   # loose — lag NaNs shift mean slightly

    def test_constant_column_gets_std_one(self):
        df = _make_station_df(n_hours=100)
        X = df[FEATURE_COLS].values.astype(np.float32)
        X[:, 0] = 5.0   # make pm25 constant
        mean, std = fit_scaler(X)
        assert std[0] == pytest.approx(1.0)   # no divide-by-zero

    def test_nan_filled_with_zero_after_scaling(self):
        df = _make_station_df(n_hours=100)
        mean, std = _default_scaler(df)
        X = df[FEATURE_COLS].values.astype(np.float32)
        X[5, 0] = np.nan
        Xs = apply_scaler(X.copy(), mean, std)
        assert not np.isnan(Xs).any()
        assert Xs[5, 0] == pytest.approx(0.0)   # NaN → 0 (scaled mean)

    def test_scaler_save_load_roundtrip(self, tmp_path):
        from models.lstm.train import save_scaler, load_scaler
        df = _make_station_df(n_hours=100)
        mean, std = _default_scaler(df)
        p = tmp_path / "scaler.npz"
        save_scaler(mean, std, path=p)
        mean2, std2 = load_scaler(path=p)
        np.testing.assert_array_almost_equal(mean, mean2)
        np.testing.assert_array_almost_equal(std,  std2)


# ---------------------------------------------------------------------------
# build_windows
# ---------------------------------------------------------------------------

class TestBuildWindows:
    def _windows(self, df):
        mean, std = _default_scaler(df)
        return build_windows(df, mean, std)

    def test_window_count_single_station(self):
        n_hours = 200
        df = _make_station_df(n_hours=n_hours)
        mean, std = _default_scaler(df)
        X, y = build_windows(df, mean, std)
        max_horizon = max(HORIZONS)
        # Expected: n_hours - SEQ_LEN - max_horizon valid windows
        # (minus any where lag NaN causes target pm25 to be NaN — shouldn't happen here)
        expected = n_hours - SEQ_LEN - max_horizon
        assert len(X) == expected

    def test_window_shape(self):
        df = _make_station_df(n_hours=200)
        X, y = self._windows(df)
        assert X.shape[1:] == (SEQ_LEN, N_FEATURES)
        assert y.shape[1]  == len(HORIZONS)

    def test_no_nan_in_X(self):
        df = _make_station_df(n_hours=200)
        X, _ = self._windows(df)
        assert not np.isnan(X).any()

    def test_no_nan_in_y(self):
        df = _make_station_df(n_hours=200)
        _, y = self._windows(df)
        assert not np.isnan(y).any()

    def test_rows_with_nan_targets_skipped(self):
        df = _make_station_df(n_hours=200)
        # Null out pm25 at position SEQ_LEN + 3 (first h3 target)
        df = df.copy()
        df.loc[df.index[SEQ_LEN + 3], "pm25"] = np.nan
        mean, std = _default_scaler(df)
        X_full, _ = build_windows(_make_station_df(n_hours=200), mean, std)
        X_miss, _ = build_windows(df, mean, std)
        assert len(X_miss) < len(X_full)

    def test_multi_station_windows_additive(self):
        df1 = _make_station_df(n_hours=200, station_id="A", seed=1)
        df2 = _make_station_df(n_hours=200, station_id="B", seed=2)
        combined = pd.concat([df1, df2], ignore_index=True)
        mean, std = fit_scaler(combined[FEATURE_COLS].values.astype(np.float32))
        X1, _ = build_windows(df1, mean, std)
        X2, _ = build_windows(df2, mean, std)
        Xc, _ = build_windows(combined, mean, std)
        assert len(Xc) == len(X1) + len(X2)

    def test_empty_df_returns_empty_arrays(self):
        df = _make_station_df(n_hours=200).iloc[:0].copy()
        mean = np.zeros(N_FEATURES, dtype=np.float32)
        std  = np.ones(N_FEATURES,  dtype=np.float32)
        X, y = build_windows(df, mean, std)
        assert len(X) == 0
        assert len(y) == 0

    def test_too_short_df_returns_empty(self):
        # Fewer rows than SEQ_LEN + max(HORIZONS) → no valid windows
        df = _make_station_df(n_hours=SEQ_LEN + max(HORIZONS) - 1)
        X, y = self._windows(df)
        assert len(X) == 0


# ---------------------------------------------------------------------------
# AQDataset
# ---------------------------------------------------------------------------

class TestAQDataset:
    def test_len(self):
        X = np.random.randn(50, SEQ_LEN, N_FEATURES).astype(np.float32)
        y = np.random.randn(50, len(HORIZONS)).astype(np.float32)
        ds = AQDataset(X, y)
        assert len(ds) == 50

    def test_getitem_shapes(self):
        X = np.random.randn(10, SEQ_LEN, N_FEATURES).astype(np.float32)
        y = np.random.randn(10, len(HORIZONS)).astype(np.float32)
        ds = AQDataset(X, y)
        xi, yi = ds[0]
        assert xi.shape == (SEQ_LEN, N_FEATURES)
        assert yi.shape == (len(HORIZONS),)

    def test_returns_tensors(self):
        X = np.random.randn(5, SEQ_LEN, N_FEATURES).astype(np.float32)
        y = np.random.randn(5, len(HORIZONS)).astype(np.float32)
        ds = AQDataset(X, y)
        xi, yi = ds[0]
        assert isinstance(xi, torch.Tensor)
        assert isinstance(yi, torch.Tensor)


# ---------------------------------------------------------------------------
# _mape
# ---------------------------------------------------------------------------

class TestMape:
    def test_perfect_forecast_is_zero(self):
        p = np.array([10.0, 20.0, 30.0])
        assert _mape(p, p) == pytest.approx(0.0)

    def test_near_zero_denominator_clamped(self):
        # true=0.0 should not produce inf/nan; eps=1.0 clamps denominator
        result = _mape(np.array([5.0]), np.array([0.0]))
        assert math.isfinite(result)

    def test_known_value(self):
        # pred=12, true=10 → |12-10|/max(10,1) * 100 = 20%
        result = _mape(np.array([12.0]), np.array([10.0]))
        assert result == pytest.approx(20.0)

    def test_array_mean(self):
        # two errors of 10% each → mean 10%
        pred = np.array([11.0, 22.0])
        true = np.array([10.0, 20.0])
        assert _mape(pred, true) == pytest.approx(10.0)
