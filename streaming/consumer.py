"""
Step 5.4–5.6 — PySpark Structured Streaming consumer.

Reads from the raw_air_quality Kafka topic, validates and engineers features,
then writes fully-processed records to the processed_air_quality topic.

Pipeline stages inside each micro-batch:
  5.4  Subscribe to Kafka → parse JSON → typed DataFrame
  5.5  Stateful feature engineering (DuckDB-assisted hybrid)
  5.6  Serialize ProcessedFeature → publish to processed_air_quality

Entry point:
  python -m streaming.consumer                        # defaults
  python -m streaming.consumer --offset earliest      # replay from beginning
  python -m streaming.consumer --checkpoint /tmp/ckpt # custom checkpoint dir
"""
from __future__ import annotations

import argparse
import os
import tempfile
from typing import Iterator

import duckdb
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType, StringType, StructField, StructType

from ingestion.database import DB_PATH
from ingestion.station_registry import load_neighbor_index, load_stations
from streaming.feature_engineering import (
    PARAMETERS,
    _add_rolling_lag_features,
    _add_temporal_features,
    _impute_series,
    assign_split,
)
from streaming.schemas import (
    ProcessedFeature,
    message_key,
    raw_reading_spark_schema,
    serialize,
)
from streaming.sensor_validation import apply_validation
from streaming.spatial_weights import compute_spatial_features

RAW_TOPIC       = "raw_air_quality"
PROCESSED_TOPIC = "processed_air_quality"
DEFAULT_BOOTSTRAP   = "localhost:9093"
DEFAULT_CHECKPOINT  = os.path.join(tempfile.gettempdir(), "aq_consumer_checkpoint")

# How many hours of DuckDB history to pull for rolling/lag context per batch.
# Rolling/lag features need up to 24 hours of past data; spatial features need
# the same window from neighbor stations.  48 hours gives a safety margin.
LOOKBACK_HOURS = 48


def _build_spark(bootstrap_servers: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("air_quality_consumer")
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        .config("spark.sql.shuffle.partitions", "19")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _kafka_stream(spark: SparkSession, bootstrap_servers: str,
                  topic: str, offset: str) -> DataFrame:
    """
    Subscribe to a Kafka topic and return a streaming DataFrame of raw bytes.

    Each row has: key (binary), value (binary), topic, partition, offset,
    timestamp, timestampType.  We use only key and value downstream.
    """
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", offset)
        .option("failOnDataLoss", "false")
        .load()
    )


def _parse_raw_messages(df: DataFrame) -> DataFrame:
    """
    Cast Kafka value bytes → JSON string → typed struct → flat columns.

    Returns one row per raw reading with columns matching RawReading fields.
    Rows that fail JSON parsing produce nulls in all parsed columns (Spark's
    from_json behaviour); these are filtered out in process_batch.
    """
    schema = raw_reading_spark_schema()
    return (
        df
        .select(F.from_json(F.col("value").cast(StringType()), schema).alias("r"))
        .select("r.*")
        .filter(F.col("station_id").isNotNull())
    )


# ---------------------------------------------------------------------------
# Batch processor (foreachBatch)
# ---------------------------------------------------------------------------

def _make_batch_processor(
    neighbor_index: dict,
    elevation_lookup: dict[str, float],
    output_bootstrap: str,
    output_topic: str,
):
    """
    Return a foreachBatch function closed over the static lookup tables.

    foreachBatch receives a micro-batch DataFrame and its batch_id.  We convert
    it to pandas, enrich with DuckDB history for rolling/lag context, run the
    same feature engineering logic as the batch pipeline, and publish results
    back to Kafka.

    Design note — DuckDB-assisted hybrid:
        Pure Spark stateful windows struggle with spatial features because
        computing a neighbor's rolling average requires the neighbor's full
        recent history, not just the current micro-batch.  Instead we:
          1. Pull the last LOOKBACK_HOURS of processed_features from DuckDB
             for all stations present in this batch.
          2. Prepend that history to the micro-batch rows.
          3. Run identical feature engineering logic from the batch pipeline.
          4. Keep only the rows whose timestamps came from the micro-batch
             (discard the prepended history rows).
        This keeps feature logic in one place and avoids duplicating the
        rolling/spatial implementation in Spark state operators.
    """
    def process_batch(batch_df: DataFrame, batch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            return

        # --- Convert micro-batch to pandas ---
        batch_pd = batch_df.toPandas()
        batch_pd["timestamp"] = pd.to_datetime(batch_pd["timestamp"], utc=True)

        # Apply sensor validation (quality_flag may already reflect AQS qualifier)
        batch_pd = apply_validation(batch_pd)
        batch_pd.loc[batch_pd["quality_flag"] >= 2, "value"] = None

        # --- Pivot micro-batch to wide format per station ---
        batch_wide: dict[str, pd.DataFrame] = {}
        for sid, sdf in batch_pd.groupby("station_id"):
            wide = (
                sdf[["timestamp", "parameter", "value"]]
                .set_index(["timestamp", "parameter"])["value"]
                .unstack("parameter")
            )
            wide.columns.name = None
            for param in PARAMETERS:
                if param not in wide.columns:
                    wide[param] = float("nan")
            batch_wide[str(sid)] = wide

        batch_timestamps = {
            str(sid): set(df.index) for sid, df in batch_wide.items()
        }

        # --- Pull DuckDB history for rolling/lag context ---
        station_ids = list(batch_wide.keys())
        min_ts = min(df.index.min() for df in batch_wide.values())
        history_from = min_ts - pd.Timedelta(hours=LOOKBACK_HOURS)

        con = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            history_sql = """
                SELECT station_id, timestamp,
                       pm25, no2, o3, pm10, co
                FROM processed_features
                WHERE station_id IN ({placeholders})
                  AND timestamp >= ?
                  AND timestamp <  ?
                ORDER BY station_id, timestamp
            """.format(placeholders=",".join("?" * len(station_ids)))

            history_pd = con.execute(
                history_sql,
                station_ids + [history_from.isoformat(), min_ts.isoformat()],
            ).df()
        finally:
            con.close()

        history_pd["timestamp"] = pd.to_datetime(history_pd["timestamp"], utc=True)

        # --- Feature engineering per station ---
        station_dfs: dict[str, pd.DataFrame] = {}

        for sid in station_ids:
            new_rows = batch_wide[sid]

            # Prepend history if available
            hist = history_pd[history_pd["station_id"] == sid].copy()
            if not hist.empty:
                hist = hist.set_index("timestamp").drop(columns=["station_id"])
                sdf = pd.concat([hist, new_rows]).sort_index()
                sdf = sdf[~sdf.index.duplicated(keep="last")]
            else:
                sdf = new_rows.sort_index()

            for param in PARAMETERS:
                sdf[param] = _impute_series(sdf[param])

            sdf = _add_temporal_features(sdf)
            sdf = _add_rolling_lag_features(sdf)
            station_dfs[sid] = sdf

        # --- Spatial features (pass 2) ---
        for sid in station_ids:
            neighbors = neighbor_index.get(sid, [])
            station_dfs[sid] = compute_spatial_features(
                sid, station_dfs[sid], neighbors, station_dfs, elevation_lookup
            )

        # --- Keep only new rows and publish ---
        from kafka import KafkaProducer  # local import — not needed until a batch arrives
        producer = KafkaProducer(
            bootstrap_servers=output_bootstrap,
            acks="all",
            retries=5,
            linger_ms=10,
            batch_size=65536,
            compression_type="lz4",
        )

        n_published = 0
        for sid, sdf in station_dfs.items():
            new_ts = batch_timestamps[sid]
            new_rows = sdf[sdf.index.isin(new_ts)]

            for ts, row in new_rows.iterrows():
                msg = ProcessedFeature(
                    station_id=sid,
                    timestamp=ts.isoformat().replace("+00:00", "Z"),
                    pm25=_nanval(row.get("pm25")),
                    no2=_nanval(row.get("no2")),
                    o3=_nanval(row.get("o3")),
                    pm10=_nanval(row.get("pm10")),
                    co=_nanval(row.get("co")),
                    hour_of_day=int(row["hour_of_day"]),
                    day_of_week=int(row["day_of_week"]),
                    month=int(row["month"]),
                    is_weekend=bool(row["is_weekend"]),
                    pm25_roll3=_nanval(row.get("pm25_roll3")),
                    pm25_roll6=_nanval(row.get("pm25_roll6")),
                    pm25_roll24=_nanval(row.get("pm25_roll24")),
                    pm25_lag1=_nanval(row.get("pm25_lag1")),
                    pm25_lag3=_nanval(row.get("pm25_lag3")),
                    pm25_lag24=_nanval(row.get("pm25_lag24")),
                    spatial_pm25_lag1=_nanval(row.get("spatial_pm25_lag1")),
                    spatial_pm25_lag3=_nanval(row.get("spatial_pm25_lag3")),
                    spatial_pm25_roll6=_nanval(row.get("spatial_pm25_roll6")),
                    spatial_no2_lag1=_nanval(row.get("spatial_no2_lag1")),
                    spatial_o3_lag1=_nanval(row.get("spatial_o3_lag1")),
                    spatial_elev_diff=_nanval(row.get("spatial_elev_diff")),
                    split=assign_split(ts),
                )
                producer.send(
                    output_topic,
                    key=message_key(sid),
                    value=serialize(msg),
                )
                n_published += 1

        producer.flush()
        producer.close()
        print(f"[batch {batch_id}] published {n_published} processed records")

    return process_batch


def _nanval(v) -> float | None:
    """Return None for NaN/None, else the float value."""
    import math
    if v is None:
        return None
    try:
        return None if math.isnan(float(v)) else float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_consumer(
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    raw_topic: str = RAW_TOPIC,
    processed_topic: str = PROCESSED_TOPIC,
    offset: str = "latest",
    checkpoint_dir: str = DEFAULT_CHECKPOINT,
) -> None:
    stations = load_stations()
    neighbor_index = load_neighbor_index()
    elevation_lookup: dict[str, float] = {
        str(s["station_id"]): float(s["elevation_m"]) if s.get("elevation_m") is not None else 0.0
        for s in stations
    }

    spark = _build_spark(bootstrap_servers)

    raw_stream = _kafka_stream(spark, bootstrap_servers, raw_topic, offset)
    parsed_stream = _parse_raw_messages(raw_stream)

    batch_fn = _make_batch_processor(
        neighbor_index, elevation_lookup,
        bootstrap_servers, processed_topic,
    )

    query = (
        parsed_stream.writeStream
        .foreachBatch(batch_fn)
        .option("checkpointLocation", checkpoint_dir)
        .trigger(processingTime="30 seconds")
        .start()
    )

    print(f"Consumer running. Reading '{raw_topic}' → writing '{processed_topic}'")
    print(f"Checkpoint: {checkpoint_dir}")
    query.awaitTermination()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    p = argparse.ArgumentParser(description="Air quality streaming consumer")
    p.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP)
    p.add_argument("--raw-topic",       default=RAW_TOPIC)
    p.add_argument("--processed-topic", default=PROCESSED_TOPIC)
    p.add_argument("--offset", default="latest",
                   choices=["latest", "earliest"],
                   help="Kafka starting offset. Use 'earliest' to replay.")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = p.parse_args()

    run_consumer(
        bootstrap_servers=args.bootstrap_servers,
        raw_topic=args.raw_topic,
        processed_topic=args.processed_topic,
        offset=args.offset,
        checkpoint_dir=args.checkpoint,
    )
