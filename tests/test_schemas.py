"""
Unit tests for streaming/schemas.py.

Tests cover:
 - RawReading and ProcessedFeature round-trip serialization
 - NaN / Inf sanitization → JSON null
 - message_key encoding
 - PySpark schema field names and counts (imported lazily; skipped if no pyspark)
"""
import json
import math

import pytest

from streaming.schemas import (
    ProcessedFeature,
    RawReading,
    deserialize,
    message_key,
    serialize,
)


# ---------------------------------------------------------------------------
# RawReading
# ---------------------------------------------------------------------------

class TestRawReadingSerialization:
    def _raw(self, **kwargs) -> RawReading:
        defaults = dict(
            station_id="06-037-1103",
            parameter="pm25",
            value=12.5,
            unit="µg/m³",
            timestamp="2024-01-15T14:00:00Z",
            quality_flag=0,
        )
        return RawReading(**{**defaults, **kwargs})

    def test_round_trip(self):
        msg = self._raw()
        payload = deserialize(serialize(msg))
        assert payload["station_id"] == "06-037-1103"
        assert payload["parameter"] == "pm25"
        assert payload["value"] == pytest.approx(12.5)
        assert payload["unit"] == "µg/m³"
        assert payload["timestamp"] == "2024-01-15T14:00:00Z"
        assert payload["quality_flag"] == 0

    def test_returns_bytes(self):
        assert isinstance(serialize(self._raw()), bytes)

    def test_valid_json(self):
        raw_bytes = serialize(self._raw())
        parsed = json.loads(raw_bytes)   # must not raise
        assert isinstance(parsed, dict)

    def test_nan_value_serialized_as_null(self):
        msg = self._raw(value=float("nan"))
        payload = deserialize(serialize(msg))
        assert payload["value"] is None

    def test_inf_value_serialized_as_null(self):
        msg = self._raw(value=float("inf"))
        payload = deserialize(serialize(msg))
        assert payload["value"] is None

    def test_all_quality_flag_values(self):
        for flag in (0, 1, 2):
            payload = deserialize(serialize(self._raw(quality_flag=flag)))
            assert payload["quality_flag"] == flag


# ---------------------------------------------------------------------------
# ProcessedFeature
# ---------------------------------------------------------------------------

class TestProcessedFeatureSerialization:
    def _feature(self, **kwargs) -> ProcessedFeature:
        defaults = dict(
            station_id="06-037-1103",
            timestamp="2024-01-15T14:00:00Z",
            pm25=12.5, no2=15.3, o3=0.045, pm10=22.1, co=0.4,
            hour_of_day=14, day_of_week=0, month=1, is_weekend=False,
            pm25_roll3=11.8, pm25_roll6=12.1, pm25_roll24=11.5,
            pm25_lag1=11.2, pm25_lag3=10.9, pm25_lag24=13.1,
            spatial_pm25_lag1=11.9, spatial_pm25_lag3=11.5,
            spatial_pm25_roll6=11.7, spatial_no2_lag1=14.8,
            spatial_o3_lag1=0.044, spatial_elev_diff=150.0,
            split="train",
        )
        return ProcessedFeature(**{**defaults, **kwargs})

    def test_round_trip_scalars(self):
        payload = deserialize(serialize(self._feature()))
        assert payload["station_id"] == "06-037-1103"
        assert payload["pm25"] == pytest.approx(12.5)
        assert payload["hour_of_day"] == 14
        assert payload["is_weekend"] is False
        assert payload["split"] == "train"

    def test_nullable_float_none_serialized_as_null(self):
        payload = deserialize(serialize(self._feature(pm25=None)))
        assert payload["pm25"] is None

    def test_nan_float_serialized_as_null(self):
        payload = deserialize(serialize(self._feature(spatial_elev_diff=float("nan"))))
        assert payload["spatial_elev_diff"] is None

    def test_all_23_feature_fields_present(self):
        payload = deserialize(serialize(self._feature()))
        expected_fields = {
            "station_id", "timestamp",
            "pm25", "no2", "o3", "pm10", "co",
            "hour_of_day", "day_of_week", "month", "is_weekend",
            "pm25_roll3", "pm25_roll6", "pm25_roll24",
            "pm25_lag1", "pm25_lag3", "pm25_lag24",
            "spatial_pm25_lag1", "spatial_pm25_lag3", "spatial_pm25_roll6",
            "spatial_no2_lag1", "spatial_o3_lag1", "spatial_elev_diff",
            "split",
        }
        assert set(payload.keys()) == expected_fields

    def test_is_weekend_true(self):
        payload = deserialize(serialize(self._feature(is_weekend=True)))
        assert payload["is_weekend"] is True


# ---------------------------------------------------------------------------
# message_key
# ---------------------------------------------------------------------------

class TestMessageKey:
    def test_returns_bytes(self):
        assert isinstance(message_key("06-037-1103"), bytes)

    def test_encodes_station_id(self):
        assert message_key("06-037-1103") == b"06-037-1103"

    def test_utf8_encoding(self):
        key = message_key("06-037-1103")
        assert key.decode("utf-8") == "06-037-1103"


# ---------------------------------------------------------------------------
# PySpark schemas (skipped if pyspark not installed)
# ---------------------------------------------------------------------------

class TestSparkSchemas:
    def test_raw_schema_field_names(self):
        pytest.importorskip("pyspark")
        from streaming.schemas import raw_reading_spark_schema
        schema = raw_reading_spark_schema()
        names = {f.name for f in schema.fields}
        assert names == {"station_id", "parameter", "value", "unit", "timestamp", "quality_flag"}

    def test_raw_schema_field_count(self):
        pytest.importorskip("pyspark")
        from streaming.schemas import raw_reading_spark_schema
        assert len(raw_reading_spark_schema().fields) == 6

    def test_processed_schema_field_names(self):
        pytest.importorskip("pyspark")
        from streaming.schemas import processed_feature_spark_schema
        from streaming.feature_engineering import SCHEMA_COLS
        schema = processed_feature_spark_schema()
        schema_names = {f.name for f in schema.fields}
        assert schema_names == set(SCHEMA_COLS)

    def test_processed_schema_field_count(self):
        pytest.importorskip("pyspark")
        from streaming.schemas import processed_feature_spark_schema
        from streaming.feature_engineering import SCHEMA_COLS
        assert len(processed_feature_spark_schema().fields) == len(SCHEMA_COLS)

    def test_nullable_fields_match_optional_dataclass(self):
        pytest.importorskip("pyspark")
        from streaming.schemas import processed_feature_spark_schema
        schema = processed_feature_spark_schema()
        # Measurement columns must be nullable (extended outages → null)
        for name in ("pm25", "no2", "o3", "pm10", "co"):
            field = next(f for f in schema.fields if f.name == name)
            assert field.nullable, f"{name} must be nullable"
        # station_id, timestamp must not be nullable
        for name in ("station_id", "timestamp"):
            field = next(f for f in schema.fields if f.name == name)
            assert not field.nullable, f"{name} must not be nullable"
