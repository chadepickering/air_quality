"""
Unit tests for streaming/consumer.py.

PySpark-specific tests are skipped if pyspark is not installed.
The batch processor is tested by injecting a mock Spark DataFrame
(whose toPandas() returns controlled pandas data) so the pandas
feature-engineering path is exercised without a running Spark session.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from streaming.consumer import _nanval, _make_batch_processor
from streaming.schemas import deserialize


# ---------------------------------------------------------------------------
# _nanval helper
# ---------------------------------------------------------------------------

class TestNanval:
    def test_none_returns_none(self):
        assert _nanval(None) is None

    def test_nan_returns_none(self):
        import math
        assert _nanval(float("nan")) is None

    def test_valid_float_returned(self):
        assert _nanval(12.5) == pytest.approx(12.5)

    def test_zero_returned(self):
        assert _nanval(0.0) == pytest.approx(0.0)

    def test_negative_returned(self):
        assert _nanval(-1.5) == pytest.approx(-1.5)

    def test_pandas_nan_returns_none(self):
        import numpy as np
        assert _nanval(np.nan) is None


# ---------------------------------------------------------------------------
# Batch processor — integration of pandas feature path
# ---------------------------------------------------------------------------

def _fake_batch_df(rows: list[dict]) -> MagicMock:
    """Return a mock Spark DataFrame whose toPandas() yields the given rows."""
    df_mock = MagicMock()
    df_mock.rdd.isEmpty.return_value = False
    df_mock.toPandas.return_value = pd.DataFrame(rows)
    return df_mock


def _minimal_rows(station_id: str = "06-037-1103",
                  n_hours: int = 3) -> list[dict]:
    """Return n_hours rows for one station, pm25 only, quality_flag=0."""
    base = pd.Timestamp("2025-01-15 10:00:00", tz="UTC")
    return [
        {
            "station_id": station_id,
            "parameter": "pm25",
            "value": 10.0 + i,
            "unit": "µg/m³",
            "timestamp": (base + pd.Timedelta(hours=i)).isoformat(),
            "quality_flag": 0,
        }
        for i in range(n_hours)
    ]


class TestBatchProcessor:
    """
    Tests for _make_batch_processor.

    DuckDB and KafkaProducer are mocked so no live services are needed.
    The test verifies that the batch processor:
      - publishes one ProcessedFeature message per new station-hour
      - each message is valid JSON with required fields
      - station_id and split labels are correctly set
      - empty batches are skipped without publishing
    """

    def _run_batch(self, rows, history_df=None):
        """Run process_batch with mocked DuckDB (empty history) and Kafka."""
        if history_df is None:
            history_df = pd.DataFrame(
                columns=["station_id", "timestamp",
                         "pm25", "no2", "o3", "pm10", "co"]
            )

        neighbor_index = {}
        elevation_lookup = {"06-037-1103": 100.0, "06-037-1104": 150.0}

        # Positional args match consumer.py signature:
        # _make_batch_processor(neighbor_index, elevation_lookup, output_bootstrap, output_topic)
        batch_fn = _make_batch_processor(
            neighbor_index, elevation_lookup,
            "localhost:9093", "processed_air_quality",
        )

        mock_producer = MagicMock()
        published = []

        def capture_send(topic, key, value):
            published.append({"topic": topic, "key": key, "value": value})
            return MagicMock()

        mock_producer.send.side_effect = capture_send

        # KafkaProducer is imported locally inside process_batch, so patch at source
        with patch("streaming.consumer.duckdb.connect") as mock_con, \
             patch("kafka.KafkaProducer", return_value=mock_producer):
            mock_con.return_value.execute.return_value.df.return_value = history_df
            mock_con.return_value.close = MagicMock()

            batch_df = _fake_batch_df(rows)
            batch_fn(batch_df, batch_id=0)

        return published

    def test_one_message_per_station_hour(self):
        rows = _minimal_rows(n_hours=3)
        published = self._run_batch(rows)
        assert len(published) == 3

    def test_published_message_is_valid_json(self):
        rows = _minimal_rows(n_hours=1)
        published = self._run_batch(rows)
        payload = deserialize(published[0]["value"])
        assert isinstance(payload, dict)

    def test_station_id_in_payload(self):
        rows = _minimal_rows(station_id="06-037-1103", n_hours=1)
        published = self._run_batch(rows)
        payload = deserialize(published[0]["value"])
        assert payload["station_id"] == "06-037-1103"

    def test_message_key_matches_station_id(self):
        rows = _minimal_rows(station_id="06-037-1103", n_hours=1)
        published = self._run_batch(rows)
        assert published[0]["key"] == b"06-037-1103"

    def test_split_field_present(self):
        rows = _minimal_rows(n_hours=1)
        published = self._run_batch(rows)
        payload = deserialize(published[0]["value"])
        assert payload["split"] in ("train", "val", "test")

    def test_all_schema_fields_present(self):
        rows = _minimal_rows(n_hours=1)
        published = self._run_batch(rows)
        payload = deserialize(published[0]["value"])
        from streaming.feature_engineering import SCHEMA_COLS
        for col in SCHEMA_COLS:
            assert col in payload, f"Missing field: {col}"

    def test_invalid_reading_value_becomes_null(self):
        # quality_flag=2 → value nulled before feature engineering
        rows = [{
            "station_id": "06-037-1103",
            "parameter": "pm25",
            "value": -999.0,
            "unit": "µg/m³",
            "timestamp": "2025-01-15T10:00:00+00:00",
            "quality_flag": 2,
        }]
        published = self._run_batch(rows)
        payload = deserialize(published[0]["value"])
        assert payload["pm25"] is None

    def test_two_stations_published_separately(self):
        rows = _minimal_rows("06-037-1103", n_hours=2) + \
               _minimal_rows("06-037-1104", n_hours=2)
        published = self._run_batch(rows)
        station_ids = {p["key"] for p in published}
        assert b"06-037-1103" in station_ids
        assert b"06-037-1104" in station_ids

    def test_empty_batch_skips_publish(self):
        df_mock = MagicMock()
        df_mock.rdd.isEmpty.return_value = True

        neighbor_index = {}
        elevation_lookup = {}
        batch_fn = _make_batch_processor(
            neighbor_index, elevation_lookup, "localhost:9093", "processed_air_quality"
        )

        mock_producer = MagicMock()
        with patch("kafka.KafkaProducer", return_value=mock_producer):
            batch_fn(df_mock, batch_id=0)

        mock_producer.send.assert_not_called()

    def test_output_topic_used_for_publish(self):
        rows = _minimal_rows(n_hours=1)
        published = self._run_batch(rows)
        assert all(p["topic"] == "processed_air_quality" for p in published)


# ---------------------------------------------------------------------------
# PySpark message parsing (skipped without pyspark)
# ---------------------------------------------------------------------------

class TestParseRawMessages:
    def test_parse_skipped_without_pyspark(self):
        pytest.importorskip("pyspark")
        # If pyspark is available, verify _parse_raw_messages returns expected schema
        from streaming.consumer import _parse_raw_messages
        # Full Spark session test belongs in integration tests (5.7)
        # Here we just confirm the function is importable and callable
        assert callable(_parse_raw_messages)
