"""
Kafka message schemas for the air quality streaming pipeline.

Two topics:
  raw_air_quality       — one message per station × parameter × hour
                          mirrors raw_readings; key = station_id (UTF-8)
  processed_air_quality — one message per station × hour (wide format)
                          mirrors processed_features; key = station_id (UTF-8)

Serialization: JSON → UTF-8 bytes.  NaN values are serialized as JSON null
so downstream consumers handle missing features without special-casing Python
float('nan'), which is not valid JSON.

PySpark StructType schemas are defined here and imported by the consumer so
the Kafka value bytes are parsed in a single from_json() call.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Python dataclasses (producer side)
# ---------------------------------------------------------------------------

@dataclass
class RawReading:
    """One hourly observation for a single parameter at a single station."""
    station_id:   str
    parameter:    str           # pm25 | no2 | o3 | pm10 | co
    value:        float
    unit:         str
    timestamp:    str           # ISO 8601 UTC: "2024-01-15T14:00:00Z"
    quality_flag: int           # 0=valid  1=suspect  2=invalid


@dataclass
class ProcessedFeature:
    """
    Full feature vector for one station-hour, ready for model inference.

    Nullable fields (Optional[float]) are NaN in the batch DB; they are
    serialized as JSON null so PySpark reads them as null in DoubleType columns.
    """
    station_id:           str
    timestamp:            str           # ISO 8601 UTC

    # Raw measurements (post-imputation; null = extended outage > 24 hr)
    pm25:                 Optional[float]
    no2:                  Optional[float]
    o3:                   Optional[float]
    pm10:                 Optional[float]
    co:                   Optional[float]

    # Temporal features
    hour_of_day:          int           # 0–23
    day_of_week:          int           # 0=Monday … 6=Sunday
    month:                int           # 1–12
    is_weekend:           bool

    # PM2.5 rolling windows (hours)
    pm25_roll3:           Optional[float]
    pm25_roll6:           Optional[float]
    pm25_roll24:          Optional[float]

    # PM2.5 lag features (hours)
    pm25_lag1:            Optional[float]
    pm25_lag3:            Optional[float]
    pm25_lag24:           Optional[float]

    # Spatial features (Epanechnikov kernel weighted average over neighbors)
    spatial_pm25_lag1:    Optional[float]
    spatial_pm25_lag3:    Optional[float]
    spatial_pm25_roll6:   Optional[float]
    spatial_no2_lag1:     Optional[float]
    spatial_o3_lag1:      Optional[float]
    spatial_elev_diff:    Optional[float]  # station_elev − weighted_neighbor_elev (m)

    # Data split label
    split:                str             # train | val | test


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _sanitize(obj):
    """Recursively replace float NaN/Inf with None so json.dumps produces null."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def serialize(msg: RawReading | ProcessedFeature) -> bytes:
    """Encode a dataclass message to UTF-8 JSON bytes for Kafka produce."""
    return json.dumps(_sanitize(asdict(msg)), separators=(",", ":")).encode("utf-8")


def deserialize(data: bytes) -> dict:
    """Decode Kafka value bytes back to a dict (used in tests and ad-hoc inspection)."""
    return json.loads(data.decode("utf-8"))


def message_key(station_id: str) -> bytes:
    """Kafka partition key: station_id as UTF-8 bytes."""
    return station_id.encode("utf-8")


# ---------------------------------------------------------------------------
# PySpark StructType schemas (consumer side)
# ---------------------------------------------------------------------------
# Imported lazily so this module can be used without PySpark installed
# (producer and tests don't depend on pyspark).

def raw_reading_spark_schema():
    """StructType for parsing raw_air_quality Kafka message values."""
    from pyspark.sql.types import (
        DoubleType, IntegerType, StringType, StructField, StructType,
    )
    return StructType([
        StructField("station_id",   StringType(),  nullable=False),
        StructField("parameter",    StringType(),  nullable=False),
        StructField("value",        DoubleType(),  nullable=True),
        StructField("unit",         StringType(),  nullable=False),
        StructField("timestamp",    StringType(),  nullable=False),
        StructField("quality_flag", IntegerType(), nullable=False),
    ])


def processed_feature_spark_schema():
    """StructType for parsing processed_air_quality Kafka message values."""
    from pyspark.sql.types import (
        BooleanType, DoubleType, IntegerType, StringType,
        StructField, StructType,
    )

    def dbl(name: str, nullable: bool = True) -> StructField:
        return StructField(name, DoubleType(), nullable=nullable)

    def int_(name: str, nullable: bool = False) -> StructField:
        return StructField(name, IntegerType(), nullable=nullable)

    return StructType([
        StructField("station_id",        StringType(),  nullable=False),
        StructField("timestamp",         StringType(),  nullable=False),
        dbl("pm25"),
        dbl("no2"),
        dbl("o3"),
        dbl("pm10"),
        dbl("co"),
        int_("hour_of_day"),
        int_("day_of_week"),
        int_("month"),
        StructField("is_weekend",        BooleanType(), nullable=False),
        dbl("pm25_roll3"),
        dbl("pm25_roll6"),
        dbl("pm25_roll24"),
        dbl("pm25_lag1"),
        dbl("pm25_lag3"),
        dbl("pm25_lag24"),
        dbl("spatial_pm25_lag1"),
        dbl("spatial_pm25_lag3"),
        dbl("spatial_pm25_roll6"),
        dbl("spatial_no2_lag1"),
        dbl("spatial_o3_lag1"),
        dbl("spatial_elev_diff"),
        StructField("split",             StringType(),  nullable=False),
    ])
