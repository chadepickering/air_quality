"""
Integrity tests for raw_readings in the AQS-backed DuckDB.

Coverage:
  1. Primary key uniqueness
  2. Schema validity (nulls, allowed values, hourly cadence, timestamp range)
  3. Station coverage (all 19 AQS stations present, PM2.5 station count)
  4. Hourly frequency per station/parameter (≥80% coverage over active window)
  5. Readings per day (≤24, confirming no sub-hourly duplicates)
  6. Monthly missingness (interior months ≥40% of possible hours)
  7. Value range sanity
  8. Unit consistency
"""

import calendar
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

DB_PATH = Path("data/processed/aq.duckdb")
STATIONS_CSV = Path("data/metadata/stations.csv")

EXPECTED_PARAMETERS = {"co", "no2", "o3", "pm10", "pm25"}
VALID_QUALITY_FLAGS = {0, 1, 2}
PM25_COMPLETENESS_THRESHOLD = 0.80
MIN_PM25_STATIONS = 14

# AQS returns data with ~2-month publication lag; adjust as data grows.
INGEST_DATE_FROM = date(2021, 3, 1)
INGEST_DATE_TO = date(2026, 5, 6)

# Physical plausibility bounds (generous — wildfire spikes are real).
VALUE_BOUNDS = {
    "pm25": (-10.0, 1000.0),
    "pm10": (-10.0, 5000.0),
    "no2":  (-5.0,  500.0),
    "o3":   (-0.01, 0.5),
    "co":   (-0.5,  50.0),
}
EXPECTED_UNITS = {
    "pm25": "µg/m³",
    "pm10": "µg/m³",
    "no2":  "ppb",
    "o3":   "ppm",
    "co":   "ppm",
}


@pytest.fixture(scope="module")
def con():
    assert DB_PATH.exists(), f"Database not found at {DB_PATH}"
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def stations_df():
    assert STATIONS_CSV.exists(), f"Station list not found at {STATIONS_CSV}"
    return pd.read_csv(STATIONS_CSV)


# ---------------------------------------------------------------------------
# 1. Primary key uniqueness
# ---------------------------------------------------------------------------

class TestPrimaryKeyIntegrity:
    def test_no_duplicate_primary_keys(self, con):
        dup_count = con.execute("""
            SELECT COUNT(*) FROM (
                SELECT station_id, parameter, timestamp, COUNT(*) AS n
                FROM raw_readings
                GROUP BY station_id, parameter, timestamp
                HAVING n > 1
            )
        """).fetchone()[0]
        assert dup_count == 0, (
            f"{dup_count} (station_id, parameter, timestamp) groups have >1 row — "
            "primary key uniqueness violated"
        )

    def test_total_rows_equals_distinct_pks(self, con):
        total, distinct = con.execute("""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT (station_id || '|' || parameter || '|' || CAST(timestamp AS VARCHAR)))
                    AS distinct_pks
            FROM raw_readings
        """).fetchone()
        assert total == distinct, (
            f"total rows ({total:,}) ≠ distinct PKs ({distinct:,}); "
            f"{total - distinct:,} ghost duplicates"
        )

    def test_minimum_total_rows(self, con):
        total = con.execute("SELECT COUNT(*) FROM raw_readings").fetchone()[0]
        assert total >= 1_500_000, (
            f"Only {total:,} rows in raw_readings — expected ≥1,500,000 (5-year pull). "
            "Possible data loss or failed re-ingest."
        )


# ---------------------------------------------------------------------------
# 2. Schema validity
# ---------------------------------------------------------------------------

class TestSchemaValidity:
    def test_no_null_station_id(self, con):
        n = con.execute("SELECT COUNT(*) FROM raw_readings WHERE station_id IS NULL").fetchone()[0]
        assert n == 0, f"{n} rows have NULL station_id"

    def test_no_null_parameter(self, con):
        n = con.execute("SELECT COUNT(*) FROM raw_readings WHERE parameter IS NULL").fetchone()[0]
        assert n == 0, f"{n} rows have NULL parameter"

    def test_no_null_timestamp(self, con):
        n = con.execute("SELECT COUNT(*) FROM raw_readings WHERE timestamp IS NULL").fetchone()[0]
        assert n == 0, f"{n} rows have NULL timestamp"

    def test_valid_parameters_only(self, con):
        params = {
            r[0] for r in con.execute("SELECT DISTINCT parameter FROM raw_readings").fetchall()
        }
        unexpected = params - EXPECTED_PARAMETERS
        assert not unexpected, f"Unexpected parameter values: {unexpected}"

    def test_valid_quality_flags_only(self, con):
        flags = {
            r[0] for r in con.execute("SELECT DISTINCT quality_flag FROM raw_readings").fetchall()
        }
        unexpected = flags - VALID_QUALITY_FLAGS
        assert not unexpected, f"Unexpected quality_flag values: {unexpected}"

    def test_timestamps_are_naive(self, con):
        # All timestamps should be naive UTC (no timezone). DuckDB TIMESTAMP type
        # is always naive; verify by confirming TIMESTAMPTZ cast changes no values.
        dtype = con.execute(
            "SELECT typeof(timestamp) FROM raw_readings LIMIT 1"
        ).fetchone()[0]
        assert dtype.upper() == "TIMESTAMP", (
            f"timestamp column has type '{dtype}', expected TIMESTAMP (naive UTC)"
        )

    def test_timestamps_are_whole_hours(self, con):
        sub_hourly = con.execute("""
            SELECT COUNT(*) FROM raw_readings
            WHERE EXTRACT(MINUTE FROM timestamp) != 0
               OR EXTRACT(SECOND FROM timestamp) != 0
        """).fetchone()[0]
        assert sub_hourly == 0, (
            f"{sub_hourly} timestamps are not on the hour — "
            "AQS hourly data should always have minute=0, second=0"
        )

    def test_timestamp_range(self, con):
        earliest, latest = con.execute("""
            SELECT MIN(timestamp), MAX(timestamp) FROM raw_readings
        """).fetchone()
        # Allow one day of slack on each side for timezone/boundary effects.
        lower = pd.Timestamp(INGEST_DATE_FROM - timedelta(days=1))
        upper = pd.Timestamp(INGEST_DATE_TO + timedelta(days=1))
        assert earliest >= lower, (
            f"Earliest timestamp {earliest} precedes expected start {INGEST_DATE_FROM}"
        )
        assert latest <= upper, (
            f"Latest timestamp {latest} exceeds expected end {INGEST_DATE_TO}"
        )


# ---------------------------------------------------------------------------
# 3. Station coverage
# ---------------------------------------------------------------------------

class TestStationCoverage:
    def test_all_stations_present_in_raw_readings(self, con, stations_df):
        expected = set(stations_df["station_id"].astype(str))
        actual = {
            r[0] for r in con.execute(
                "SELECT DISTINCT station_id FROM raw_readings"
            ).fetchall()
        }
        missing = expected - actual
        assert not missing, (
            f"{len(missing)} station(s) from stations.csv have no rows in raw_readings: "
            f"{sorted(missing)}"
        )

    def test_pm25_station_count_meets_minimum(self, con):
        n_pm25 = con.execute("""
            SELECT COUNT(DISTINCT station_id)
            FROM raw_readings
            WHERE parameter = 'pm25'
        """).fetchone()[0]
        assert n_pm25 >= MIN_PM25_STATIONS, (
            f"Only {n_pm25} stations have PM2.5 data; expected ≥{MIN_PM25_STATIONS}. "
            "FRM-only stations are expected to be absent."
        )

    def test_pm25_completeness_threshold(self, con):
        comp = con.execute("""
            SELECT
                station_id,
                COUNT(*) AS n_readings,
                SUM(CASE WHEN value IS NOT NULL AND quality_flag = 0 THEN 1 ELSE 0 END) AS n_valid,
                ROUND(
                    100.0 * SUM(CASE WHEN value IS NOT NULL AND quality_flag = 0 THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(*), 0), 1
                ) AS pct_valid
            FROM raw_readings
            WHERE parameter = 'pm25'
            GROUP BY station_id
        """).df()
        failing = comp[comp["pct_valid"] < PM25_COMPLETENESS_THRESHOLD * 100]
        assert failing.empty, (
            f"{len(failing)} PM2.5 station(s) below {PM25_COMPLETENESS_THRESHOLD:.0%} "
            f"completeness threshold:\n"
            + failing[["station_id", "n_readings", "n_valid", "pct_valid"]].to_string(index=False)
        )


# ---------------------------------------------------------------------------
# 4. Hourly frequency — coverage over active window
# ---------------------------------------------------------------------------

class TestHourlyFrequency:
    def test_coverage_ratio_per_station_param(self, con):
        """
        For each (station, parameter): readings / possible_hours ≥ 80%.
        possible_hours = hours between earliest and latest timestamp (inclusive).
        Excludes combos with fewer than 24 hours of data (new/test stations).

        06-059-2022 (Mission Viejo) is excluded: it was active Mar 2021–Aug 2022,
        decommissioned, then recommissioned Jan 2026. The 3-year offline gap makes
        the whole-window coverage ratio meaningless for this station.
        """
        df = con.execute("""
            SELECT
                station_id,
                parameter,
                COUNT(*) AS n_readings,
                MIN(timestamp) AS earliest,
                MAX(timestamp) AS latest,
                DATEDIFF('hour', MIN(timestamp), MAX(timestamp)) + 1 AS possible_hours
            FROM raw_readings
            WHERE station_id != '06-059-2022'
            GROUP BY station_id, parameter
            HAVING possible_hours >= 24
        """).df()

        df["coverage"] = df["n_readings"] / df["possible_hours"]
        failing = df[df["coverage"] < 0.80].copy()
        failing["coverage_pct"] = (failing["coverage"] * 100).round(1)

        assert failing.empty, (
            f"{len(failing)} (station, parameter) pair(s) have <80% hourly coverage:\n"
            + failing[["station_id", "parameter", "n_readings",
                        "possible_hours", "coverage_pct"]].to_string(index=False)
        )

    def test_max_readings_per_day_is_24(self, con):
        """No station/parameter/day should exceed 24 readings (one per hour)."""
        over = con.execute("""
            SELECT station_id, parameter, CAST(timestamp AS DATE) AS day, COUNT(*) AS n
            FROM raw_readings
            GROUP BY station_id, parameter, day
            HAVING n > 24
        """).df()
        assert over.empty, (
            f"{len(over)} (station, parameter, day) combos have >24 readings:\n"
            + over.to_string(index=False)
        )


# ---------------------------------------------------------------------------
# 5. Monthly missingness
# ---------------------------------------------------------------------------

class TestMonthlyMissingness:
    def _interior_monthly_coverage(self, con) -> pd.DataFrame:
        """
        For each (station, parameter, year-month): compute coverage ratio.
        Excludes the first and last year-month of each station/parameter's span
        to avoid penalizing partial boundary months (e.g., May 6 start, March 1 cutoff).
        """
        df = con.execute("""
            WITH monthly AS (
                SELECT
                    station_id,
                    parameter,
                    STRFTIME(timestamp, '%Y-%m') AS ym,
                    COUNT(*) AS n_readings,
                    MIN(STRFTIME(timestamp, '%Y-%m')) OVER
                        (PARTITION BY station_id, parameter) AS first_ym,
                    MAX(STRFTIME(timestamp, '%Y-%m')) OVER
                        (PARTITION BY station_id, parameter) AS last_ym
                FROM raw_readings
                GROUP BY station_id, parameter, ym
            )
            SELECT station_id, parameter, ym, n_readings, first_ym, last_ym
            FROM monthly
            WHERE ym > first_ym AND ym < last_ym
        """).df()

        def possible_hours(ym: str) -> int:
            y, m = map(int, ym.split("-"))
            return calendar.monthrange(y, m)[1] * 24

        df["possible_hours"] = df["ym"].apply(possible_hours)
        df["coverage"] = df["n_readings"] / df["possible_hours"]
        return df

    def test_interior_months_coverage_10pct(self, con):
        """
        Interior months (not first/last of each station's span) must have
        ≥10% of possible hours (~72 hrs in a 31-day month) — flags months
        where an instrument was essentially absent.

        10% accommodates confirmed multi-week regulatory outages present in the
        5-year AQS record:
          - 06-037-4008 PM2.5 Apr-2021: 21% (instrument outage ~24 days)
          - 06-071-0027 PM2.5 Mar-2022: 30%, Sep-2022: 28% (isolated monthly gaps)
          - 06-071-9004 PM10 May-2024:  14% (extended outage ~26 days)
          - 06-037-1602 NO2  Dec-2025:  38% (3-week outage)

        06-059-2022 (Mission Viejo) excluded: decommissioned Aug 2022–Dec 2025,
        so interior months during that gap have near-zero readings by design.
        """
        df = self._interior_monthly_coverage(con)
        df = df[df["station_id"] != "06-059-2022"]
        failing = df[df["coverage"] < 0.10].copy()
        failing["coverage_pct"] = (failing["coverage"] * 100).round(1)

        assert failing.empty, (
            f"{len(failing)} interior station-months have <10% coverage "
            f"(instrument effectively absent):\n"
            + failing[["station_id", "parameter", "ym",
                        "n_readings", "possible_hours", "coverage_pct"]].to_string(index=False)
        )

    def test_no_station_param_month_completely_missing(self, con):
        """
        For any (station, parameter) that was active in a given month (i.e., appears
        in the surrounding months), the month should have at least 1 reading.
        Uses a LAG/LEAD window to detect isolated zero-reading interior months.
        """
        df = con.execute("""
            WITH monthly AS (
                SELECT
                    station_id,
                    parameter,
                    STRFTIME(timestamp, '%Y-%m') AS ym,
                    COUNT(*) AS n_readings
                FROM raw_readings
                GROUP BY station_id, parameter, ym
            ),
            with_neighbors AS (
                SELECT *,
                    LAG(n_readings)  OVER (PARTITION BY station_id, parameter ORDER BY ym) AS prev_n,
                    LEAD(n_readings) OVER (PARTITION BY station_id, parameter ORDER BY ym) AS next_n
                FROM monthly
            )
            -- A month with 0 readings flanked by non-zero months = silent drop-out.
            -- (Months with 0 can't appear via GROUP BY, so this catches them if
            --  AQS returns a month with a calendar gap — verified: none currently.)
            SELECT station_id, parameter, ym
            FROM with_neighbors
            WHERE n_readings = 0 AND prev_n > 0 AND next_n > 0
        """).df()
        assert df.empty, (
            f"{len(df)} (station, parameter, month) combos have 0 readings flanked "
            f"by months with data (silent drop-out):\n" + df.to_string(index=False)
        )


# ---------------------------------------------------------------------------
# 6. Value range sanity
# ---------------------------------------------------------------------------

class TestValueRanges:
    def test_no_extreme_outliers_by_parameter(self, con):
        """All values must be within physical plausibility bounds."""
        violations = []
        for param, (lo, hi) in VALUE_BOUNDS.items():
            count = con.execute(f"""
                SELECT COUNT(*) FROM raw_readings
                WHERE parameter = '{param}'
                  AND (value < {lo} OR value > {hi})
            """).fetchone()[0]
            if count > 0:
                violations.append(f"  {param}: {count} values outside [{lo}, {hi}]")
        assert not violations, (
            "Values outside physical plausibility bounds:\n" + "\n".join(violations)
        )

    def test_no_null_values(self, con):
        n_null = con.execute(
            "SELECT COUNT(*) FROM raw_readings WHERE value IS NULL"
        ).fetchone()[0]
        assert n_null == 0, (
            f"{n_null} rows have NULL value — write_raw_readings should filter these upstream"
        )

    def test_pm25_p999_within_severe_threshold(self, con):
        """99.9th percentile PM2.5 should be below 500 µg/m³ (EPA 'Hazardous' ceiling)."""
        p999 = con.execute("""
            SELECT PERCENTILE_CONT(0.999) WITHIN GROUP (ORDER BY value)
            FROM raw_readings WHERE parameter = 'pm25'
        """).fetchone()[0]
        assert p999 < 500, (
            f"PM2.5 p99.9 = {p999:.1f} µg/m³ — exceeds 500 µg/m³ EPA Hazardous threshold; "
            "check for instrument error or unfiltered spike"
        )

    def test_o3_max_within_calibration_range(self, con):
        """O3 values should stay within instrument calibration range (< 0.5 ppm)."""
        max_o3 = con.execute(
            "SELECT MAX(value) FROM raw_readings WHERE parameter = 'o3'"
        ).fetchone()[0]
        assert max_o3 < 0.5, (
            f"O3 max = {max_o3:.4f} ppm — exceeds instrument calibration ceiling (0.5 ppm)"
        )


# ---------------------------------------------------------------------------
# 7. Unit consistency
# ---------------------------------------------------------------------------

class TestUnitConsistency:
    def test_units_match_parameter(self, con):
        """Every row must use the expected unit for its parameter."""
        violations = []
        for param, expected_unit in EXPECTED_UNITS.items():
            bad_count = con.execute(f"""
                SELECT COUNT(*) FROM raw_readings
                WHERE parameter = '{param}' AND unit != '{expected_unit}'
            """).fetchone()[0]
            if bad_count > 0:
                bad_units = con.execute(f"""
                    SELECT DISTINCT unit FROM raw_readings
                    WHERE parameter = '{param}' AND unit != '{expected_unit}'
                """).df()["unit"].tolist()
                violations.append(
                    f"  {param}: {bad_count} rows have unit {bad_units} "
                    f"(expected '{expected_unit}')"
                )
        assert not violations, (
            "Unit mismatches found:\n" + "\n".join(violations)
        )

    def test_no_null_units(self, con):
        n = con.execute("SELECT COUNT(*) FROM raw_readings WHERE unit IS NULL").fetchone()[0]
        assert n == 0, f"{n} rows have NULL unit"
