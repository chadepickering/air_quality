"""
Unit tests for streaming/spatial_weights.py.

Tests cover:
 - _weighted_spatial_avg: NaN handling, weight renormalization, edge cases
 - compute_spatial_features: isolated stations, single neighbor, multi-neighbor,
   static elevation, column completeness, neighbor weight invariant
"""
import numpy as np
import pandas as pd
import pytest

from streaming.spatial_weights import _weighted_spatial_avg, compute_spatial_features


def _make_idx(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", name="timestamp")


def _make_df(n: int = 48, seed: int = 0, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = _make_idx(n, start)
    return pd.DataFrame(
        {
            "pm25":      rng.uniform(5,  50, n),
            "no2":       rng.uniform(5,  40, n),
            "o3":        rng.uniform(0.01, 0.1, n),
            "pm25_roll6": rng.uniform(5,  50, n),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# _weighted_spatial_avg
# ---------------------------------------------------------------------------

class TestWeightedSpatialAvg:
    def _s(self, values, start="2024-01-01"):
        return pd.Series(values, index=_make_idx(len(values), start))

    def test_single_neighbor_unit_weight(self):
        s = self._s([1.0, 2.0, 3.0])
        result = _weighted_spatial_avg([s], [1.0])
        pd.testing.assert_series_equal(result.reset_index(drop=True),
                                       s.reset_index(drop=True))

    def test_equal_weights_two_neighbors(self):
        s1 = self._s([2.0, 4.0])
        s2 = self._s([4.0, 6.0])
        result = _weighted_spatial_avg([s1, s2], [0.5, 0.5])
        expected = self._s([3.0, 5.0])
        pd.testing.assert_series_equal(result.reset_index(drop=True),
                                       expected.reset_index(drop=True))

    def test_unequal_weights(self):
        s1 = self._s([0.0])
        s2 = self._s([10.0])
        result = _weighted_spatial_avg([s1, s2], [0.25, 0.75])
        assert result.iloc[0] == pytest.approx(7.5)

    def test_nan_in_one_neighbor_renormalizes_weight(self):
        s1 = self._s([np.nan, 4.0])
        s2 = self._s([2.0,    6.0])
        result = _weighted_spatial_avg([s1, s2], [0.5, 0.5])
        # t=0: only s2 valid, normalized weight = 1.0 -> result = 2.0
        # t=1: both valid -> (0.5*4 + 0.5*6) / 1.0 = 5.0
        assert result.iloc[0] == pytest.approx(2.0)
        assert result.iloc[1] == pytest.approx(5.0)

    def test_all_nan_returns_nan(self):
        s1 = self._s([np.nan, np.nan])
        s2 = self._s([np.nan, np.nan])
        result = _weighted_spatial_avg([s1, s2], [0.5, 0.5])
        assert result.isna().all()

    def test_empty_list_returns_empty_series(self):
        result = _weighted_spatial_avg([], [])
        assert len(result) == 0

    def test_non_unit_weights_normalized_implicitly(self):
        # Weights don't have to sum to 1 in the raw call -- renormalization handles it
        s1 = self._s([2.0])
        s2 = self._s([4.0])
        result_raw   = _weighted_spatial_avg([s1, s2], [3.0, 3.0])
        result_normed = _weighted_spatial_avg([s1, s2], [0.5, 0.5])
        assert result_raw.iloc[0] == pytest.approx(result_normed.iloc[0])


# ---------------------------------------------------------------------------
# compute_spatial_features
# ---------------------------------------------------------------------------

class TestComputeSpatialFeatures:
    SPATIAL_COLS = [
        "spatial_pm25_lag1", "spatial_pm25_lag3", "spatial_pm25_roll6",
        "spatial_no2_lag1",  "spatial_o3_lag1",   "spatial_elev_diff",
    ]

    def test_isolated_station_all_spatial_nan(self):
        df = _make_df()
        result = compute_spatial_features("s1", df, [], {}, {})
        for col in self.SPATIAL_COLS:
            assert result[col].isna().all(), f"{col} must be NaN for isolated station"

    def test_spatial_cols_always_added(self):
        df = _make_df()
        result = compute_spatial_features("s1", df, [], {}, {})
        for col in self.SPATIAL_COLS:
            assert col in result.columns

    def test_original_df_not_mutated(self):
        df = _make_df()
        original_cols = set(df.columns)
        _ = compute_spatial_features("s1", df, [], {}, {})
        assert set(df.columns) == original_cols

    def test_single_neighbor_pm25_lag1_correct(self):
        n = 48
        s1_df = _make_df(n, seed=1)
        s2_df = _make_df(n, seed=2)
        all_dfs = {"s1": s1_df, "s2": s2_df}
        elev = {"s1": 100.0, "s2": 200.0}
        neighbors = [("s2", 1.0)]

        result = compute_spatial_features("s1", s1_df, neighbors, all_dfs, elev)

        expected = s2_df["pm25"].shift(1)
        pd.testing.assert_series_equal(
            result["spatial_pm25_lag1"],
            expected,
            check_names=False,
        )

    def test_single_neighbor_pm25_lag3_correct(self):
        n = 48
        s1_df = _make_df(n, seed=1)
        s2_df = _make_df(n, seed=2)
        all_dfs = {"s1": s1_df, "s2": s2_df}
        elev = {"s1": 0.0, "s2": 0.0}
        neighbors = [("s2", 1.0)]

        result = compute_spatial_features("s1", s1_df, neighbors, all_dfs, elev)

        expected = s2_df["pm25"].shift(3)
        pd.testing.assert_series_equal(
            result["spatial_pm25_lag3"],
            expected,
            check_names=False,
        )

    def test_static_elev_diff_single_neighbor(self):
        s1_df = _make_df()
        s2_df = _make_df()
        all_dfs = {"s1": s1_df, "s2": s2_df}
        elev = {"s1": 100.0, "s2": 400.0}
        neighbors = [("s2", 1.0)]

        result = compute_spatial_features("s1", s1_df, neighbors, all_dfs, elev)

        # weight=1.0 * |100 - 400| = 300.0 for all rows
        assert result["spatial_elev_diff"].sub(300.0).abs().max() < 1e-9

    def test_static_elev_diff_two_neighbors(self):
        s1_df = _make_df(seed=0)
        s2_df = _make_df(seed=1)
        s3_df = _make_df(seed=2)
        all_dfs = {"s1": s1_df, "s2": s2_df, "s3": s3_df}
        elev = {"s1": 0.0, "s2": 100.0, "s3": 200.0}
        # weights already normalized: 0.6 + 0.4 = 1.0
        neighbors = [("s2", 0.6), ("s3", 0.4)]

        result = compute_spatial_features("s1", s1_df, neighbors, all_dfs, elev)

        expected_elev_diff = 0.6 * abs(0.0 - 100.0) + 0.4 * abs(0.0 - 200.0)
        assert result["spatial_elev_diff"].sub(expected_elev_diff).abs().max() < 1e-9

    def test_missing_neighbor_in_all_station_dfs_skipped(self):
        s1_df = _make_df()
        # s2 is listed as a neighbor but absent from all_station_dfs
        all_dfs = {"s1": s1_df}
        elev = {"s1": 0.0, "s2": 0.0}
        neighbors = [("s2", 1.0)]

        result = compute_spatial_features("s1", s1_df, neighbors, all_dfs, elev)

        # All spatial cols should be NaN since the only neighbor is absent
        for col in self.SPATIAL_COLS:
            assert result[col].isna().all(), f"{col} should be NaN"

    def test_spatial_roll6_uses_precomputed_column(self):
        n = 48
        s1_df = _make_df(n, seed=0)
        # Build a neighbor df with known pm25_roll6 values
        s2_df = _make_df(n, seed=1)
        sentinel_roll6 = np.full(n, 42.0)
        s2_df["pm25_roll6"] = sentinel_roll6

        all_dfs = {"s1": s1_df, "s2": s2_df}
        elev = {"s1": 0.0, "s2": 0.0}
        neighbors = [("s2", 1.0)]

        result = compute_spatial_features("s1", s1_df, neighbors, all_dfs, elev)

        # spatial_pm25_roll6 should equal the sentinel 42.0 (from pre-computed column)
        assert result["spatial_pm25_roll6"].sub(42.0).abs().max() < 1e-9

    def test_neighbor_weights_sum_to_one_in_index(self):
        from ingestion.station_registry import build_spatial_neighbor_index, load_stations
        try:
            stations = load_stations()
        except FileNotFoundError:
            pytest.skip("stations.csv not available in this environment")

        # Convert to list of dicts with float elevation
        stations_clean = [
            s for s in stations
            if s.get("elevation_m") is not None
        ]
        index = build_spatial_neighbor_index(stations_clean)

        for sid, neighbors in index.items():
            if neighbors:
                total_weight = sum(w for _, w in neighbors)
                assert total_weight == pytest.approx(1.0, abs=1e-9), (
                    f"Neighbor weights for {sid} sum to {total_weight}, expected 1.0"
                )
