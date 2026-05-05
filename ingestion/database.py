from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = "data/processed/aq.duckdb"


def initialize_database(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Create DuckDB schema tables if they don't exist; return open connection."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_readings (
            station_id   VARCHAR,
            parameter    VARCHAR,
            value        FLOAT,
            unit         VARCHAR,
            timestamp    TIMESTAMP,
            quality_flag INTEGER DEFAULT 0,
            ingested_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (station_id, parameter, timestamp)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS processed_features (
            station_id         VARCHAR,
            timestamp          TIMESTAMP,
            pm25               FLOAT,
            no2                FLOAT,
            o3                 FLOAT,
            pm10               FLOAT,
            co                 FLOAT,
            hour_of_day        INTEGER,
            day_of_week        INTEGER,
            month              INTEGER,
            is_weekend         BOOLEAN,
            pm25_roll3         FLOAT,
            pm25_roll6         FLOAT,
            pm25_roll24        FLOAT,
            pm25_lag1          FLOAT,
            pm25_lag3          FLOAT,
            pm25_lag24         FLOAT,
            spatial_pm25_lag1  FLOAT,
            spatial_pm25_lag3  FLOAT,
            spatial_pm25_roll6 FLOAT,
            spatial_no2_lag1   FLOAT,
            spatial_o3_lag1    FLOAT,
            spatial_elev_diff  FLOAT,
            PRIMARY KEY (station_id, timestamp)
        )
    """)
    return con


def write_raw_readings(con: duckdb.DuckDBPyConnection, readings: list[dict]) -> int:
    """
    Insert raw sensor readings, skipping duplicates (INSERT OR IGNORE).
    quality_flag: 0=valid, 1=suspect, 2=invalid

    Returns number of rows in the input batch.
    """
    if not readings:
        return 0
    df = pd.DataFrame(readings)
    if "quality_flag" not in df.columns:
        df["quality_flag"] = 0
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    con.execute("""
        INSERT OR IGNORE INTO raw_readings
            (station_id, parameter, value, unit, timestamp, quality_flag)
        SELECT station_id, parameter, value, unit, timestamp, quality_flag
        FROM df
    """)
    return len(df)


def write_processed_features(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Upsert processed feature rows, skipping duplicates. Returns row count."""
    if df.empty:
        return 0
    con.execute("""
        INSERT OR IGNORE INTO processed_features
        SELECT * FROM df
    """)
    return len(df)


def get_station_completeness(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-station, per-parameter data completeness report."""
    return con.execute("""
        SELECT
            station_id,
            parameter,
            COUNT(*)                                                          AS n_readings,
            MIN(timestamp)                                                    AS earliest,
            MAX(timestamp)                                                    AS latest,
            SUM(CASE WHEN value IS NOT NULL AND quality_flag = 0 THEN 1 ELSE 0 END) AS n_valid,
            ROUND(
                100.0 * SUM(CASE WHEN value IS NOT NULL AND quality_flag = 0 THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0),
                1
            )                                                                 AS pct_valid
        FROM raw_readings
        GROUP BY station_id, parameter
        ORDER BY station_id, parameter
    """).df()
