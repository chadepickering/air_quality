"""
Unit tests for streaming/sensor_validation.py.

Tests cover:
 - validate_reading: all five parameters, both tiers, boundary behavior
 - apply_validation: vectorized flag update, no-downgrade invariant
"""
import numpy as np
import pandas as pd
import pytest

from streaming.sensor_validation import (
    INVALID_BOUNDS,
    SUSPECT_BOUNDS,
    apply_validation,
    validate_reading,
)


class TestValidateReadingBoundStructure:
    def test_all_parameters_have_bounds(self):
        for param in ["pm25", "no2", "o3", "pm10", "co"]:
            assert param in SUSPECT_BOUNDS
            assert param in INVALID_BOUNDS

    def test_invalid_bounds_strictly_wider_than_suspect(self):
        for param in SUSPECT_BOUNDS:
            s_lo, s_hi = SUSPECT_BOUNDS[param]
            i_lo, i_hi = INVALID_BOUNDS[param]
            assert i_lo <= s_lo, f"{param} invalid lower must be <= suspect lower"
            assert i_hi >= s_hi, f"{param} invalid upper must be >= suspect upper"


class TestValidateReadingPM25:
    def test_typical_valid(self):
        assert validate_reading(15.0, "pm25") == 0

    def test_zero_valid(self):
        # 0.0 is the suspect lower bound — exactly at floor is valid
        assert validate_reading(0.0, "pm25") == 0

    def test_small_negative_suspect(self):
        # Any negative is physically impossible -> suspect regardless of magnitude.
        # EPA guidance: retain these (not imputed) to avoid upward bias in averages.
        assert validate_reading(-1.0, "pm25") == 1
        assert validate_reading(-5.0, "pm25") == 1
        assert validate_reading(-7.7, "pm25") == 1   # 5yr observed minimum

    def test_extreme_negative_invalid(self):
        # Far below observed noise floor (-7.7) -> instrument malfunction
        assert validate_reading(-100.0, "pm25") == 2

    def test_wildfire_peak_valid(self):
        # 400 ug/m3 is within suspect bounds (ceiling = 500) -> valid
        assert validate_reading(400.0, "pm25") == 0

    def test_above_suspect_ceiling_is_suspect(self):
        _, s_hi = SUSPECT_BOUNDS["pm25"]
        assert validate_reading(s_hi + 1.0, "pm25") == 1

    def test_extreme_high_invalid(self):
        assert validate_reading(10000.0, "pm25") == 2

    def test_at_invalid_floor_boundary(self):
        i_lo, _ = INVALID_BOUNDS["pm25"]
        assert validate_reading(i_lo - 0.001, "pm25") == 2


class TestValidateReadingNO2:
    def test_typical_valid(self):
        assert validate_reading(25.0, "no2") == 0

    def test_small_negative_suspect(self):
        # All negatives are physically impossible -> suspect
        assert validate_reading(-0.3, "no2") == 1   # 5yr observed median negative
        assert validate_reading(-1.7, "no2") == 1   # 5yr observed minimum

    def test_extreme_negative_invalid(self):
        assert validate_reading(-200.0, "no2") == 2

    def test_scaqmd_5yr_max_valid(self):
        # 5yr SCAQMD max ~95 ppb -- within suspect ceiling (500)
        assert validate_reading(95.0, "no2") == 0


class TestValidateReadingO3:
    def test_typical_valid(self):
        assert validate_reading(0.05, "o3") == 0

    def test_tiny_negative_suspect(self):
        # AQS dataset has readings down to -0.004 ppm — physically impossible, retained as suspect
        assert validate_reading(-0.001, "o3") == 1
        assert validate_reading(-0.004, "o3") == 1   # 5yr observed minimum

    def test_extreme_negative_invalid(self):
        # Far below observed noise floor (-0.004 ppm) -> invalid
        assert validate_reading(-0.1, "o3") == 2

    def test_5yr_max_valid(self):
        # 5yr LA metro max 0.145 ppm -- within suspect ceiling (0.5)
        assert validate_reading(0.145, "o3") == 0

    def test_above_suspect_ceiling_is_suspect(self):
        # 5.0 ppm is above suspect ceiling (0.5) but below invalid ceiling (10.0)
        assert validate_reading(5.0, "o3") == 1

    def test_extreme_invalid(self):
        assert validate_reading(15.0, "o3") == 2


class TestValidateReadingPM10:
    def test_typical_valid(self):
        assert validate_reading(50.0, "pm10") == 0

    def test_coachella_dust_storm_valid(self):
        # 2000 ug/m3 is within suspect ceiling (3000) -- valid, not suspect
        assert validate_reading(2000.0, "pm10") == 0

    def test_above_suspect_ceiling_suspect(self):
        _, s_hi = SUSPECT_BOUNDS["pm10"]
        assert validate_reading(s_hi + 1.0, "pm10") == 1

    def test_extreme_high_invalid(self):
        assert validate_reading(100000.0, "pm10") == 2


class TestValidateReadingCO:
    def test_typical_valid(self):
        assert validate_reading(0.5, "co") == 0

    def test_near_road_max_valid(self):
        # Near-road 5yr max ~10 ppm -- within suspect ceiling (20)
        assert validate_reading(10.0, "co") == 0

    def test_above_suspect_ceiling_suspect(self):
        _, s_hi = SUSPECT_BOUNDS["co"]
        assert validate_reading(s_hi + 1.0, "co") == 1

    def test_instrument_negative_suspect(self):
        # AQS dataset CO min is -0.2 ppm -- negative, so suspect (physically impossible)
        assert validate_reading(-0.1, "co") == 1
        assert validate_reading(-0.2, "co") == 1   # 5yr observed minimum

    def test_invalid_negative(self):
        # Below invalid floor (-1.0) -> instrument malfunction
        assert validate_reading(-2.0, "co") == 2


class TestValidateReadingUnknownParameter:
    def test_unknown_param_returns_valid(self):
        # No bounds defined -> default (-1e9, 1e9) -> always valid
        assert validate_reading(999999.0, "sulfur_dioxide") == 0


class TestApplyValidation:
    def _df(self, rows: list[tuple]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["value", "parameter", "quality_flag"])

    def test_valid_readings_unchanged(self):
        df = self._df([(15.0, "pm25", 0), (30.0, "no2", 0)])
        result = apply_validation(df)
        assert list(result["quality_flag"]) == [0, 0]

    def test_suspect_reading_flag_set(self):
        # Any negative PM2.5 is physically impossible -> suspect
        df = self._df([(-1.0, "pm25", 0)])
        result = apply_validation(df)
        assert result["quality_flag"].iloc[0] == 1

    def test_invalid_reading_flag_set(self):
        df = self._df([(-100.0, "pm25", 0)])
        result = apply_validation(df)
        assert result["quality_flag"].iloc[0] == 2

    def test_existing_flag_not_downgraded(self):
        # AQS qualifier already suspect (1); range check says valid (0) -- keep 1
        df = self._df([(15.0, "pm25", 1)])
        result = apply_validation(df)
        assert result["quality_flag"].iloc[0] == 1

    def test_existing_flag_upgraded_to_invalid(self):
        # AQS qualifier suspect (1); range says invalid (2) -- upgrade to 2
        df = self._df([(-100.0, "pm25", 1)])
        result = apply_validation(df)
        assert result["quality_flag"].iloc[0] == 2

    def test_original_dataframe_not_mutated(self):
        df = self._df([(-100.0, "pm25", 0)])
        _ = apply_validation(df)
        assert df["quality_flag"].iloc[0] == 0

    def test_mixed_parameters_vectorized(self):
        df = self._df([
            (15.0,  "pm25", 0),   # valid
            (-1.0,  "pm25", 0),   # suspect (negative -> physically impossible)
            (-100.0,"pm25", 0),   # invalid (below invalid floor -15.0)
            (30.0,  "no2",  0),   # valid
            (-0.3,  "no2",  0),   # suspect (negative)
        ])
        result = apply_validation(df)
        assert list(result["quality_flag"]) == [0, 1, 2, 0, 1]

    def test_all_parameters_normal_range_valid(self):
        rows = [
            (15.0,  "pm25", 0),
            (25.0,  "no2",  0),
            (0.05,  "o3",   0),
            (50.0,  "pm10", 0),
            (0.5,   "co",   0),
        ]
        df = self._df(rows)
        result = apply_validation(df)
        assert (result["quality_flag"] == 0).all()
