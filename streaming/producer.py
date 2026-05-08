"""
Step 5.3 — Kafka producer: historical replay from raw_readings.

Reads rows from the raw_readings DuckDB table and publishes them to the
raw_air_quality Kafka topic, one message per station × parameter × hour.

Messages are produced in ascending timestamp order so that each station
partition receives records in chronological sequence.  Key = station_id
(UTF-8 bytes), which pins all readings for a given station to the same
partition and preserves per-station temporal order.

Replay speed:
  --rate N   Messages per second.  0 (default) = unlimited (as fast as the
             broker accepts).  Use 0 for integration tests and batch backfill.
             For real-time simulation of 14 FEM stations × 5 parameters, the
             natural arrival rate is 70 messages/hour ≈ 0.019 msg/s.

Entry point:
  python -m streaming.producer                             # full history, unlimited
  python -m streaming.producer --date-from 2025-01-01 --date-to 2025-01-31
  python -m streaming.producer --rate 500                  # 500 msg/s throttle
"""
from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timezone

import duckdb
from kafka import KafkaProducer
from kafka.errors import KafkaError

from ingestion.database import DB_PATH
from streaming.schemas import RawReading, message_key, serialize

TOPIC = "raw_air_quality"
DEFAULT_BOOTSTRAP = "localhost:9093"
REPORT_EVERY = 10_000   # print a progress line every N messages


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay raw_readings into Kafka")
    p.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP)
    p.add_argument("--topic", default=TOPIC)
    p.add_argument(
        "--date-from", default=None,
        help="Start date inclusive (YYYY-MM-DD). Default: earliest in DB.",
    )
    p.add_argument(
        "--date-to", default=None,
        help="End date inclusive (YYYY-MM-DD). Default: latest in DB.",
    )
    p.add_argument(
        "--rate", type=float, default=0.0,
        help="Max messages per second. 0 = unlimited.",
    )
    return p.parse_args()


def _build_query(date_from: str | None, date_to: str | None) -> tuple[str, list]:
    """Return (sql, params) for the raw_readings fetch."""
    clauses, params = [], []
    if date_from:
        clauses.append("timestamp >= ?")
        params.append(f"{date_from}T00:00:00Z")
    if date_to:
        clauses.append("timestamp <= ?")
        params.append(f"{date_to}T23:59:59Z")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT station_id, parameter, value, unit, timestamp, quality_flag
        FROM raw_readings
        {where}
        ORDER BY timestamp, station_id, parameter
    """
    return sql, params


def run_producer(
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    topic: str = TOPIC,
    date_from: str | None = None,
    date_to: str | None = None,
    rate: float = 0.0,
) -> int:
    """
    Fetch rows from raw_readings and publish to Kafka.

    Returns the total number of messages produced.
    """
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        acks="all",           # wait for leader + ISR acknowledgment
        retries=5,
        linger_ms=10,         # small batching window to improve throughput
        batch_size=65536,     # 64 KB batch
        compression_type="lz4",
    )

    con = duckdb.connect(str(DB_PATH), read_only=True)
    sql, params = _build_query(date_from, date_to)

    print(f"Querying raw_readings ({date_from or 'all'} → {date_to or 'all'})...")
    rows = con.execute(sql, params).fetchall()
    con.close()

    total = len(rows)
    print(f"  {total:,} rows to publish → topic '{topic}' on {bootstrap_servers}")

    min_interval = 1.0 / rate if rate > 0 else 0.0
    errors: list[str] = []
    produced = 0
    t_start = time.monotonic()

    for row in rows:
        station_id, parameter, value, unit, timestamp, quality_flag = row

        msg = RawReading(
            station_id=str(station_id),
            parameter=str(parameter),
            value=float(value) if value is not None else float("nan"),
            unit=str(unit),
            timestamp=str(timestamp),
            quality_flag=int(quality_flag),
        )

        future = producer.send(
            topic,
            key=message_key(msg.station_id),
            value=serialize(msg),
        )
        future.add_errback(lambda e, sid=station_id, ts=timestamp:
                           errors.append(f"{sid}@{ts}: {e}"))

        produced += 1

        if produced % REPORT_EVERY == 0:
            elapsed = time.monotonic() - t_start
            rate_actual = produced / elapsed if elapsed > 0 else 0
            pct = 100 * produced / total
            print(f"  {produced:>8,} / {total:,} ({pct:.1f}%)  {rate_actual:.0f} msg/s")

        if min_interval > 0:
            time.sleep(min_interval)

    producer.flush()
    elapsed = time.monotonic() - t_start
    rate_actual = produced / elapsed if elapsed > 0 else 0
    print(f"Done. {produced:,} messages in {elapsed:.1f}s ({rate_actual:.0f} msg/s)")

    if errors:
        print(f"WARNING: {len(errors)} delivery errors:")
        for e in errors[:10]:
            print(f"  {e}")

    return produced


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    args = _parse_args()
    run_producer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        date_from=args.date_from,
        date_to=args.date_to,
        rate=args.rate,
    )
