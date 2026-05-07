from datetime import datetime, timedelta, timezone
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
    Insert raw sensor readings, skipping duplicates on PRIMARY KEY conflict.
    quality_flag: 0=valid, 1=suspect, 2=invalid

    Returns number of rows in the input batch.
    """
    if not readings:
        return 0
    df = pd.DataFrame(readings)
    if "quality_flag" not in df.columns:
        df["quality_flag"] = 0
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    con.execute("""
        INSERT INTO raw_readings
            (station_id, parameter, value, unit, timestamp, quality_flag)
        SELECT station_id, parameter, value, unit, timestamp, quality_flag
        FROM df
        ON CONFLICT DO NOTHING
    """)
    return len(df)


def write_processed_features(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Insert processed feature rows, skipping duplicates. Returns row count."""
    if df.empty:
        return 0
    con.execute("""
        INSERT INTO processed_features
        SELECT * FROM df
        ON CONFLICT DO NOTHING
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


PM25_COMPLETENESS_THRESHOLD = 80.0


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from ingestion.openaq_client import fetch_measurements, load_sensor_index

    sensor_index = load_sensor_index()
    print(f"Loaded sensor index: {len(sensor_index)} sensors.")

    con = initialize_database()
    print(f"DuckDB initialised at {DB_PATH}.")

    now = datetime.now(timezone.utc)

    items = list(sensor_index.items())
    total_written = 0
    skipped = 0
    for i, (sensor_id, meta) in enumerate(items):
        # Use the sensor's last-known report date as date_to so stale sensors
        # still pull their actual last year of data.
        date_last_str = meta.get("date_last", "")
        if date_last_str:
            date_to = min(
                datetime.fromisoformat(date_last_str.replace("Z", "+00:00")),
                now,
            )
        else:
            date_to = now
        date_from = date_to - timedelta(days=365)

        # Skip sensors with no data expected (last report > 3 years ago)
        if (now - date_to).days > 3 * 365:
            print(
                f"  [{i+1:>3}/{len(items)}] sensor {sensor_id:>8} "
                f"({meta['station_id']} / {meta['parameter']}): skipped (last report {date_last_str[:10]})"
            )
            skipped += 1
            continue

        try:
            raw = fetch_measurements(sensor_id, date_from, date_to)
            readings = [
                {
                    "station_id": meta["station_id"],
                    "parameter":  meta["parameter"],
                    "value":      r["value"],
                    "unit":       meta["unit"],
                    "timestamp":  r["timestamp"],
                }
                for r in raw
            ]
            n = write_raw_readings(con, readings)
            total_written += n
            print(
                f"  [{i+1:>3}/{len(items)}] sensor {sensor_id:>8} "
                f"({meta['station_id']} / {meta['parameter']}): {n} rows"
            )
        except Exception as e:
            print(f"  [{i+1:>3}/{len(items)}] sensor {sensor_id} FAILED: {e}")

    print(f"\nTotal rows written: {total_written:,} ({skipped} sensors skipped — last report >3 years ago)")

    print("\n--- Completeness report ---")
    comp = get_station_completeness(con)
    print(comp.to_string(index=False))

    pm25 = comp[comp["parameter"] == "pm25"].copy()
    flagged = pm25[pm25["pct_valid"] < PM25_COMPLETENESS_THRESHOLD]
    if flagged.empty:
        print(f"\nAll stations meet the {PM25_COMPLETENESS_THRESHOLD}% PM2.5 completeness threshold.")
    else:
        print(f"\nStations below {PM25_COMPLETENESS_THRESHOLD}% PM2.5 completeness (flagged for exclusion):")
        print(flagged[["station_id", "n_readings", "n_valid", "pct_valid"]].to_string(index=False))

    con.close()
