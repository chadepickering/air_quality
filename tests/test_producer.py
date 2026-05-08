"""
Unit tests for streaming/producer.py.

Kafka broker calls are mocked — these tests verify query construction,
message serialization correctness, and rate-limiting logic without
requiring a running broker or DuckDB file.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from streaming.producer import _build_query, run_producer
from streaming.schemas import deserialize


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

class TestBuildQuery:
    def test_no_filters_has_no_where(self):
        sql, params = _build_query(None, None)
        assert "WHERE" not in sql
        assert params == []

    def test_date_from_only(self):
        sql, params = _build_query("2025-01-01", None)
        assert "WHERE" in sql
        assert "timestamp >=" in sql
        assert params == ["2025-01-01T00:00:00Z"]

    def test_date_to_only(self):
        sql, params = _build_query(None, "2025-01-31")
        assert "WHERE" in sql
        assert "timestamp <=" in sql
        assert params == ["2025-01-31T23:59:59Z"]

    def test_both_dates(self):
        sql, params = _build_query("2025-01-01", "2025-01-31")
        assert "timestamp >=" in sql
        assert "timestamp <=" in sql
        assert len(params) == 2

    def test_order_by_timestamp_station_parameter(self):
        sql, _ = _build_query(None, None)
        assert "ORDER BY timestamp, station_id, parameter" in sql


# ---------------------------------------------------------------------------
# run_producer — message shape and delivery
# ---------------------------------------------------------------------------

def _make_rows(n: int = 3) -> list[tuple]:
    """Return n fake raw_readings rows."""
    return [
        (f"06-037-{1000 + i}", "pm25", 12.5 + i, "µg/m³",
         f"2025-01-01T{i:02d}:00:00Z", 0)
        for i in range(n)
    ]


class TestRunProducer:
    def _patched(self, rows, mock_producer):
        """Context: DuckDB returns `rows`; KafkaProducer is replaced by mock."""
        mock_future = MagicMock()
        mock_future.add_errback = MagicMock()
        mock_producer.return_value.send.return_value = mock_future
        return mock_future

    @patch("streaming.producer.KafkaProducer")
    @patch("streaming.producer.duckdb.connect")
    def test_produces_one_message_per_row(self, mock_connect, mock_kp):
        rows = _make_rows(5)
        mock_connect.return_value.execute.return_value.fetchall.return_value = rows
        self._patched(rows, mock_kp)

        n = run_producer()

        assert n == 5
        assert mock_kp.return_value.send.call_count == 5

    @patch("streaming.producer.KafkaProducer")
    @patch("streaming.producer.duckdb.connect")
    def test_message_key_is_station_id_bytes(self, mock_connect, mock_kp):
        rows = _make_rows(1)
        mock_connect.return_value.execute.return_value.fetchall.return_value = rows
        self._patched(rows, mock_kp)

        run_producer()

        _, kwargs = mock_kp.return_value.send.call_args
        assert kwargs["key"] == b"06-037-1000"

    @patch("streaming.producer.KafkaProducer")
    @patch("streaming.producer.duckdb.connect")
    def test_message_value_is_valid_json_with_correct_fields(self, mock_connect, mock_kp):
        rows = _make_rows(1)
        mock_connect.return_value.execute.return_value.fetchall.return_value = rows
        self._patched(rows, mock_kp)

        run_producer()

        _, kwargs = mock_kp.return_value.send.call_args
        payload = deserialize(kwargs["value"])
        assert payload["station_id"] == "06-037-1000"
        assert payload["parameter"] == "pm25"
        assert payload["value"] == pytest.approx(12.5)
        assert payload["unit"] == "µg/m³"
        assert payload["quality_flag"] == 0

    @patch("streaming.producer.KafkaProducer")
    @patch("streaming.producer.duckdb.connect")
    def test_null_value_serialized_as_json_null(self, mock_connect, mock_kp):
        rows = [("06-037-1000", "pm25", None, "µg/m³", "2025-01-01T00:00:00Z", 0)]
        mock_connect.return_value.execute.return_value.fetchall.return_value = rows
        self._patched(rows, mock_kp)

        run_producer()

        _, kwargs = mock_kp.return_value.send.call_args
        payload = deserialize(kwargs["value"])
        assert payload["value"] is None

    @patch("streaming.producer.KafkaProducer")
    @patch("streaming.producer.duckdb.connect")
    def test_flush_called_after_all_sends(self, mock_connect, mock_kp):
        rows = _make_rows(3)
        mock_connect.return_value.execute.return_value.fetchall.return_value = rows
        self._patched(rows, mock_kp)

        run_producer()

        mock_kp.return_value.flush.assert_called_once()

    @patch("streaming.producer.KafkaProducer")
    @patch("streaming.producer.duckdb.connect")
    def test_topic_passed_to_send(self, mock_connect, mock_kp):
        rows = _make_rows(1)
        mock_connect.return_value.execute.return_value.fetchall.return_value = rows
        self._patched(rows, mock_kp)

        run_producer(topic="custom_topic")

        args, _ = mock_kp.return_value.send.call_args
        assert args[0] == "custom_topic"

    @patch("streaming.producer.KafkaProducer")
    @patch("streaming.producer.duckdb.connect")
    def test_empty_db_produces_zero_messages(self, mock_connect, mock_kp):
        mock_connect.return_value.execute.return_value.fetchall.return_value = []
        self._patched([], mock_kp)

        n = run_producer()

        assert n == 0
        mock_kp.return_value.send.assert_not_called()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    @patch("streaming.producer.time.sleep")
    @patch("streaming.producer.KafkaProducer")
    @patch("streaming.producer.duckdb.connect")
    def test_rate_zero_does_not_sleep(self, mock_connect, mock_kp, mock_sleep):
        rows = _make_rows(3)
        mock_connect.return_value.execute.return_value.fetchall.return_value = rows
        mock_future = MagicMock()
        mock_future.add_errback = MagicMock()
        mock_kp.return_value.send.return_value = mock_future

        run_producer(rate=0.0)

        mock_sleep.assert_not_called()

    @patch("streaming.producer.time.sleep")
    @patch("streaming.producer.KafkaProducer")
    @patch("streaming.producer.duckdb.connect")
    def test_rate_positive_sleeps_between_messages(self, mock_connect, mock_kp, mock_sleep):
        rows = _make_rows(3)
        mock_connect.return_value.execute.return_value.fetchall.return_value = rows
        mock_future = MagicMock()
        mock_future.add_errback = MagicMock()
        mock_kp.return_value.send.return_value = mock_future

        run_producer(rate=10.0)   # 10 msg/s → sleep 0.1s between each

        assert mock_sleep.call_count == 3
        for c in mock_sleep.call_args_list:
            assert c == call(pytest.approx(0.1, rel=1e-3))
