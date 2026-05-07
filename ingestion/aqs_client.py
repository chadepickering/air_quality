import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://aqs.epa.gov/data/api"

LA_BBOX = {
    "lat_min": 33.5, "lat_max": 34.8,
    "lon_min": -118.9, "lon_max": -117.0,
}

SCAQMD_COUNTIES = [
    {"state": "06", "county": "037"},  # Los Angeles
    {"state": "06", "county": "059"},  # Orange
    {"state": "06", "county": "065"},  # Riverside
    {"state": "06", "county": "071"},  # San Bernardino
    {"state": "06", "county": "111"},  # Ventura
]

# AQS parameter codes and the unit string stored in raw_readings.
# Note: AQS returns native units (ppb for NO2, ppm for O3/CO) — not converted to µg/m³.
# Sensor validation ranges in Step 4 must match these units.
AQS_PARAMETERS = {
    "pm25": {"code": "88101", "unit": "µg/m³"},
    "no2":  {"code": "42602", "unit": "ppb"},
    "o3":   {"code": "44201", "unit": "ppm"},
    "pm10": {"code": "81102", "unit": "µg/m³"},
    "co":   {"code": "42101", "unit": "ppm"},
}

METADATA_DIR = Path("data/metadata")


def _auth() -> dict:
    return {"email": os.getenv("AQS_EMAIL"), "key": os.getenv("AQS_KEY")}


def _in_bbox(lat: float, lon: float) -> bool:
    return (
        LA_BBOX["lat_min"] <= lat <= LA_BBOX["lat_max"]
        and LA_BBOX["lon_min"] <= lon <= LA_BBOX["lon_max"]
    )


# ---------------------------------------------------------------------------
# Station discovery
# ---------------------------------------------------------------------------

def fetch_monitors_by_county(state: str, county: str, param_code: str) -> list[dict]:
    """
    GET /monitors/byCounty — returns one row per instrument (POC) per site.

    Uses a fixed recent year for the date filter so only monitors active within
    the last two years are returned. Elevation is included in the response and
    is complete for all LA metro SCAQMD stations (no USGS EPQS call needed).
    """
    resp = requests.get(
        f"{BASE_URL}/monitors/byCounty",
        params={
            **_auth(),
            "param": param_code,
            "bdate": "20240101",
            "edate": "20251231",
            "state": state,
            "county": county,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("Data", [])


def build_station_list() -> pd.DataFrame:
    """
    Discover all active PM2.5 continuous monitors across SCAQMD counties within
    the LA metro bbox.

    Queries monitors/byCounty for PM2.5 (88101), deduplicates by site
    (multiple POCs at the same physical site → one row), excludes closed
    monitors, and filters to the LA metro bounding box.

    Returns DataFrame with columns:
        station_id, name, lat, lon, elevation_m, county_code, state_code
    """
    seen: dict[str, dict] = {}

    for c in SCAQMD_COUNTIES:
        monitors = fetch_monitors_by_county(c["state"], c["county"], "88101")
        for m in monitors:
            lat = m.get("latitude")
            lon = m.get("longitude")
            if lat is None or lon is None:
                continue
            if not _in_bbox(lat, lon):
                continue
            if m.get("close_date"):
                continue
            sid = f"{m['state_code']}-{m['county_code']}-{m['site_number']}"
            if sid not in seen:
                seen[sid] = {
                    "station_id":  sid,
                    "name":        m.get("local_site_name") or m.get("address", ""),
                    "lat":         lat,
                    "lon":         lon,
                    "elevation_m": float(m["elevation"]) if m.get("elevation") is not None else None,
                    "county_code": m["county_code"],
                    "state_code":  m["state_code"],
                }
        time.sleep(0.5)

    df = pd.DataFrame(list(seen.values()))
    return df.sort_values("station_id").reset_index(drop=True)


def save_station_list(df: pd.DataFrame, path: Path | None = None) -> None:
    path = path or METADATA_DIR / "stations.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} stations to {path}")


# ---------------------------------------------------------------------------
# Historical data pull
# ---------------------------------------------------------------------------

def _qualifier_to_flag(qualifier) -> int:
    """Blank qualifier → 0 (valid). Any non-blank qualifier string → 1 (suspect)."""
    if not qualifier:
        return 0
    return 0 if str(qualifier).strip() == "" else 1


def fetch_samples_by_county(
    param_code: str,
    state: str,
    county: str,
    date_from: date,
    date_to: date,
) -> list[dict]:
    """
    GET /sampleData/byCounty — all raw sample rows for one county/parameter/period.

    Returns only 1-HOUR duration rows (filters out 24-hr FRM filter readings).
    """
    resp = requests.get(
        f"{BASE_URL}/sampleData/byCounty",
        params={
            **_auth(),
            "param": param_code,
            "bdate": date_from.strftime("%Y%m%d"),
            "edate": date_to.strftime("%Y%m%d"),
            "state": state,
            "county": county,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return [
        r for r in resp.json().get("Data", [])
        if r.get("sample_duration") == "1 HOUR"
    ]


def fetch_historical_all(
    station_ids: set[str],
    date_from: date,
    date_to: date,
    chunk_days: int = 90,
) -> list[dict]:
    """
    Fetch one year of hourly data across all SCAQMD counties and all five parameters.

    Strategy: 5 counties × 5 parameters × ~4 quarterly chunks ≈ 100 requests total,
    completing in ~5 minutes. Each county-level request returns all stations at once —
    no per-sensor iteration.

    Where multiple instruments (POC values) report the same parameter at the same
    site and hour, keeps the lowest POC with a non-null measurement value.

    AQS timestamp convention: time_gmt is the start of the 1-hour averaging period.
    Stored as UTC ISO string: "{date_gmt}T{time_gmt}:00Z".

    Returns flat list of dicts matching raw_readings schema:
        station_id, parameter, value, unit, timestamp, quality_flag
    """
    all_readings: list[dict] = []
    request_count = 0

    for param_name, param_info in AQS_PARAMETERS.items():
        param_code = param_info["code"]
        unit = param_info["unit"]

        for county_info in SCAQMD_COUNTIES:
            state, county = county_info["state"], county_info["county"]
            chunk_start = date_from

            while chunk_start <= date_to:
                # AQS sampleData requires bdate and edate within the same calendar year.
                year_end = date(chunk_start.year, 12, 31)
                chunk_end = min(
                    chunk_start + timedelta(days=chunk_days - 1),
                    year_end,
                    date_to,
                )
                request_count += 1

                try:
                    rows = fetch_samples_by_county(
                        param_code, state, county, chunk_start, chunk_end
                    )
                except requests.HTTPError as e:
                    print(
                        f"  [{request_count:>3}] WARN {param_name} {state}-{county} "
                        f"{chunk_start}–{chunk_end}: HTTP {e.response.status_code} — skipped"
                    )
                    chunk_start = chunk_end + timedelta(days=1)
                    time.sleep(1.0)
                    continue

                # Per (station, timestamp): keep lowest POC with non-null measurement.
                best: dict[tuple[str, str], dict] = {}
                for r in rows:
                    sid = f"{r['state_code']}-{r['county_code']}-{r['site_number']}"
                    if sid not in station_ids:
                        continue
                    value = r.get("sample_measurement")
                    if value is None:
                        continue
                    ts = f"{r['date_gmt']}T{r['time_gmt']}:00Z"
                    poc = int(r.get("poc") or 99)
                    key = (sid, ts)
                    existing = best.get(key)
                    if existing is None or poc < existing["_poc"]:
                        best[key] = {
                            "station_id":   sid,
                            "parameter":    param_name,
                            "value":        float(value),
                            "unit":         unit,
                            "timestamp":    ts,
                            "quality_flag": _qualifier_to_flag(r.get("qualifier")),
                            "_poc":         poc,
                        }

                chunk_readings = [
                    {k: v for k, v in rec.items() if k != "_poc"}
                    for rec in best.values()
                ]
                all_readings.extend(chunk_readings)
                print(
                    f"  [{request_count:>3}] {param_name:>4} {state}-{county} "
                    f"{chunk_start}–{chunk_end}: {len(chunk_readings):>5} readings"
                )
                chunk_start = chunk_end + timedelta(days=1)
                time.sleep(0.5)

    return all_readings


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    import sys
    from ingestion.database import initialize_database, write_raw_readings, get_station_completeness

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Station discovery ---
    print("Fetching PM2.5 monitor list from AQS for all SCAQMD counties...")
    df_stations = build_station_list()
    print(f"Found {len(df_stations)} active PM2.5 stations within LA metro bbox.")
    save_station_list(df_stations)
    print(df_stations[["station_id", "name", "lat", "lon", "elevation_m"]].to_string(index=False))

    station_ids = set(df_stations["station_id"])

    # --- Historical data pull ---
    date_to   = date.today()
    date_from = date(2021, 3, 1)
    print(f"\nFetching hourly sample data {date_from} → {date_to} "
          f"for {len(station_ids)} stations across 5 parameters...")

    con = initialize_database()
    readings = fetch_historical_all(station_ids, date_from, date_to)
    n = write_raw_readings(con, readings)
    print(f"\nTotal rows written to DuckDB: {n:,}")

    # --- Completeness report ---
    print("\n--- Completeness report ---")
    comp = get_station_completeness(con)
    print(comp.to_string(index=False))

    pm25 = comp[comp["parameter"] == "pm25"].copy()
    passing = pm25[pm25["pct_valid"] >= 80.0]
    flagged = pm25[pm25["pct_valid"] < 80.0]
    print(f"\nPM2.5 stations meeting 80% completeness: {len(passing)}")
    if not flagged.empty:
        print(f"Stations below 80% PM2.5 completeness (flagged for exclusion):")
        print(flagged[["station_id", "n_readings", "n_valid", "pct_valid"]].to_string(index=False))

    con.close()
