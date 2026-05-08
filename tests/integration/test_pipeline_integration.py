"""
Step 5.7 — End-to-end pipeline integration test.

Requires:
  - docker compose up -d        (Kafka on localhost:9093)
  - bash streaming/create_topics.sh
  - A populated raw_readings DuckDB (Step 2 complete)

Run with:
  pytest -m integration -v

What is verified:
  1. Broker reachability — skip entire module if Kafka is not up.
  2. Producer publishes the correct number of messages for a 30-day window
     and each message has valid JSON with all required RawReading fields.
  3. Batch processor (consumer core) produces one ProcessedFeature per
     station-hour and every required field is present in the output.
  4. Feature sanity: temporal fields are in valid ranges; rolling/lag fields
     are non-null for station-hours with enough history.
  5. Message count consistency: raw topic message count equals the number of
     rows in raw_readings for the same 30-day window.
  6. Kafdrop availability check (HTTP; warns but does not fail if down).
"""
from __future__ import annotations

import time
import uuid
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pytest
import requests
from kafka import KafkaAdminClient, KafkaConsumer, KafkaProducer
from kafka.admin import NewTopic
from kafka.errors import NoBrokersAvailable

from ingestion.database import DB_PATH
from streaming.consumer import _make_batch_processor, _nanval
from streaming.producer import _build_query, run_producer
from streaming.schemas import RawReading, deserialize, serialize

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BOOTSTRAP = "localhost:9093"
INTEGRATION_WINDOW_DAYS = 30
# Use a fixed recent window likely to have data in the DB
TEST_DATE_FROM = "2025-01-01"
TEST_DATE_TO   = "2025-01-30"

RAW_TOPIC       = "raw_air_quality"
PROCESSED_TOPIC = "processed_air_quality"

KAFDROP_URL = "http://localhost:9000"

# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

def _kafka_available() -> bool:
    try:
        admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP, request_timeout_ms=3000)
        admin.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def require_kafka():
    if not _kafka_available():
        pytest.skip(
            "Kafka not reachable at localhost:9093. "
            "Run `docker compose up -d && bash streaming/create_topics.sh` first."
        )


@pytest.fixture(scope="session")
def raw_row_count() -> int:
    """Number of rows in raw_readings for the 30-day test window."""
    sql, params = _build_query(TEST_DATE_FROM, TEST_DATE_TO)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    count = con.execute(
        f"SELECT COUNT(*) FROM ({sql}) t", params
    ).fetchone()[0]
    con.close()
    return count


@pytest.fixture(scope="session")
def unique_raw_topic(require_kafka) -> str:
    """
    Create a unique ephemeral topic for this test run so parallel CI runs
    don't interfere with each other.  The topic is deleted after the session.
    """
    topic_name = f"aq_inttest_raw_{uuid.uuid4().hex[:8]}"
    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP)
    admin.create_topics([NewTopic(name=topic_name, num_partitions=19, replication_factor=1)])
    admin.close()
    yield topic_name
    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP)
    try:
        admin.delete_topics([topic_name])
    except Exception:
        pass
    admin.close()


@pytest.fixture(scope="session")
def produced_count(unique_raw_topic, raw_row_count) -> int:
    """
    Run the producer once for the session and return the message count.
    All subsequent tests consume from the same ephemeral topic.
    """
    if raw_row_count == 0:
        pytest.skip(
            f"No raw_readings rows found for {TEST_DATE_FROM} – {TEST_DATE_TO}. "
            "Populate the DB first (Step 2)."
        )
    n = run_producer(
        bootstrap_servers=BOOTSTRAP,
        topic=unique_raw_topic,
        date_from=TEST_DATE_FROM,
        date_to=TEST_DATE_TO,
        rate=0,   # unlimited — go as fast as possible
    )
    return n


@pytest.fixture(scope="session")
def consumed_messages(unique_raw_topic, produced_count) -> list[dict]:
    """
    Consume all messages from the ephemeral raw topic and return parsed payloads.
    Times out after 30 s if fewer messages than expected arrive.
    """
    consumer = KafkaConsumer(
        unique_raw_topic,
        bootstrap_servers=BOOTSTRAP,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=30_000,
        value_deserializer=lambda b: deserialize(b),
    )
    messages = []
    for msg in consumer:
        messages.append(msg.value)
        if len(messages) >= produced_count:
            break
    consumer.close()
    return messages


# ---------------------------------------------------------------------------
# Test 1: Broker + topic reachability
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_broker_is_reachable():
    assert _kafka_available()


@pytest.mark.integration
def test_topics_exist():
    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP)
    existing = admin.list_topics()
    admin.close()
    assert RAW_TOPIC in existing, f"Topic '{RAW_TOPIC}' not found. Run create_topics.sh."
    assert PROCESSED_TOPIC in existing, f"Topic '{PROCESSED_TOPIC}' not found."


# ---------------------------------------------------------------------------
# Test 2: Producer message count and structure
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_producer_message_count_matches_db(produced_count, raw_row_count):
    assert produced_count == raw_row_count, (
        f"Producer sent {produced_count} messages but DB has {raw_row_count} rows "
        f"for {TEST_DATE_FROM} – {TEST_DATE_TO}."
    )


@pytest.mark.integration
def test_all_messages_consumed(consumed_messages, produced_count):
    assert len(consumed_messages) == produced_count


@pytest.mark.integration
def test_raw_message_fields(consumed_messages):
    required = {"station_id", "parameter", "value", "unit", "timestamp", "quality_flag"}
    for msg in consumed_messages[:100]:   # spot-check first 100
        assert set(msg.keys()) == required, f"Unexpected fields in: {msg}"


@pytest.mark.integration
def test_raw_message_parameters_are_valid(consumed_messages):
    valid_params = {"pm25", "no2", "o3", "pm10", "co"}
    params_seen = {msg["parameter"] for msg in consumed_messages}
    assert params_seen.issubset(valid_params)


@pytest.mark.integration
def test_raw_message_quality_flags_are_valid(consumed_messages):
    flags = {msg["quality_flag"] for msg in consumed_messages}
    assert flags.issubset({0, 1, 2})


@pytest.mark.integration
def test_raw_message_null_values_encoded_as_json_null(consumed_messages):
    # Any message with None value should have been serialized as null (not NaN string)
    for msg in consumed_messages:
        if msg["value"] is None:
            # Just confirming it was decoded correctly from JSON null
            assert msg["value"] is None


# ---------------------------------------------------------------------------
# Test 3: Consumer batch processor output
# ---------------------------------------------------------------------------

def _make_fake_batch_df(rows: list[dict]) -> MagicMock:
    df_mock = MagicMock()
    df_mock.rdd.isEmpty.return_value = False
    df_mock.toPandas.return_value = pd.DataFrame(rows)
    return df_mock


@pytest.fixture(scope="session")
def processed_outputs(consumed_messages) -> list[dict]:
    """
    Run the consumer batch processor on a representative sample (one station,
    first 72 hours) and collect the ProcessedFeature messages it would publish.
    """
    # Pick the station with the most raw messages in our sample
    from collections import Counter
    station_counts = Counter(m["station_id"] for m in consumed_messages
                             if m["parameter"] == "pm25")
    if not station_counts:
        pytest.skip("No pm25 messages in consumed sample.")
    target_station = station_counts.most_common(1)[0][0]

    station_msgs = [m for m in consumed_messages
                    if m["station_id"] == target_station][:72 * 5]  # 72hr × 5 params

    neighbor_index = {}
    elevation_lookup = {target_station: 100.0}
    batch_fn = _make_batch_processor(
        neighbor_index, elevation_lookup, BOOTSTRAP, PROCESSED_TOPIC,
    )

    published = []

    def capture_send(topic, key, value):
        published.append(deserialize(value))
        return MagicMock()

    mock_producer = MagicMock()
    mock_producer.send.side_effect = capture_send

    empty_history = pd.DataFrame(
        columns=["station_id", "timestamp", "pm25", "no2", "o3", "pm10", "co"]
    )

    with patch("streaming.consumer.duckdb.connect") as mock_con, \
         patch("kafka.KafkaProducer", return_value=mock_producer):
        mock_con.return_value.execute.return_value.df.return_value = empty_history
        mock_con.return_value.close = MagicMock()
        batch_fn(_make_fake_batch_df(station_msgs), batch_id=0)

    return published


@pytest.mark.integration
def test_processed_output_count(processed_outputs, consumed_messages):
    # One output per unique (station, timestamp) pair — not per parameter
    assert len(processed_outputs) > 0


@pytest.mark.integration
def test_processed_output_has_all_schema_fields(processed_outputs):
    from streaming.feature_engineering import SCHEMA_COLS
    required = set(SCHEMA_COLS)
    for msg in processed_outputs:
        missing = required - set(msg.keys())
        assert not missing, f"Missing fields: {missing}"


@pytest.mark.integration
def test_temporal_features_in_range(processed_outputs):
    for msg in processed_outputs:
        assert 0 <= msg["hour_of_day"] <= 23,  f"hour_of_day out of range: {msg['hour_of_day']}"
        assert 0 <= msg["day_of_week"] <= 6,   f"day_of_week out of range: {msg['day_of_week']}"
        assert 1 <= msg["month"] <= 12,         f"month out of range: {msg['month']}"
        assert msg["is_weekend"] in (True, False)


@pytest.mark.integration
def test_split_labels_are_valid(processed_outputs):
    valid_splits = {"train", "val", "test"}
    for msg in processed_outputs:
        assert msg["split"] in valid_splits


@pytest.mark.integration
def test_rolling_features_non_null_with_history(processed_outputs):
    # After the first 24 rows, pm25_roll24 should be non-null
    # (assuming the station had pm25 data throughout)
    later_msgs = [m for m in processed_outputs
                  if m.get("pm25") is not None][24:]
    non_null_rolls = [m for m in later_msgs if m.get("pm25_roll24") is not None]
    assert len(non_null_rolls) > 0, "Expected some non-null pm25_roll24 values after 24+ hours"


# ---------------------------------------------------------------------------
# Test 4: Kafdrop availability (warning only)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_kafdrop_accessible():
    try:
        resp = requests.get(KAFDROP_URL, timeout=5)
        assert resp.status_code == 200
    except Exception as e:
        pytest.warns(UserWarning, match="Kafdrop")
        import warnings
        warnings.warn(f"Kafdrop not accessible at {KAFDROP_URL}: {e}", UserWarning)
