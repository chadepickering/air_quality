import os
import time
from datetime import datetime, timedelta, timezone

import requests

BASE_URL = "https://api.openaq.org/v3"

# South Coast AQMD bounding box: min_lon,min_lat,max_lon,max_lat
LA_BBOX = "-118.9,33.5,-117.0,34.8"
LA_PARAMETERS = ["pm25", "no2", "o3", "pm10", "co"]

# OpenAQ v3 parameter IDs (used when filtering by ID is more reliable than name)
PARAM_IDS = {"pm25": 2, "no2": 7, "o3": 9, "pm10": 1, "co": 6}


def _headers() -> dict:
    api_key = os.getenv("OPENAQ_API_KEY")
    return {"X-API-Key": api_key} if api_key else {}


def fetch_la_stations() -> list[dict]:
    """Fetch all PM2.5-capable stations in the LA metro bounding box, paginated."""
    stations = []
    page = 1
    while True:
        params = {
            "bbox": LA_BBOX,
            "parameters_id": PARAM_IDS["pm25"],
            "limit": 100,
            "page": page,
        }
        resp = requests.get(f"{BASE_URL}/locations", params=params, headers=_headers(), timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        stations.extend(results)
        if len(results) < 100:
            break
        page += 1
        time.sleep(0.5)
    return stations


def parse_station_metadata(raw_stations: list[dict]) -> list[dict]:
    """Flatten OpenAQ location objects to minimal station dicts for stations.csv."""
    out = []
    for loc in raw_stations:
        coords = loc.get("coordinates") or {}
        lat = coords.get("latitude")
        lon = coords.get("longitude")
        if lat is None or lon is None:
            continue
        out.append({
            "station_id": str(loc["id"]),
            "name": loc.get("name", ""),
            "lat": lat,
            "lon": lon,
            "country": (loc.get("country") or {}).get("code", ""),
            "timezone": loc.get("timezone", ""),
        })
    return out


def fetch_measurements(
    location_id: str,
    parameter: str,
    date_from: datetime,
    date_to: datetime,
    page_limit: int = 1000,
) -> list[dict]:
    """Fetch all measurements for one station/parameter over a date range, paginated."""
    readings = []
    page = 1
    while True:
        params = {
            "locations_id": location_id,
            "parameters_name": parameter,
            "date_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_to": date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": page_limit,
            "page": page,
        }
        resp = requests.get(
            f"{BASE_URL}/measurements", params=params, headers=_headers(), timeout=30
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        readings.extend(results)
        if len(results) < page_limit:
            break
        page += 1
        time.sleep(0.3)
    return readings


def fetch_historical_bulk(
    stations: list[dict],
    parameters: list[str] | None = None,
    lookback_days: int = 365,
) -> list[dict]:
    """
    Pull historical measurements for all stations and parameters.
    Returns flat list of dicts ready for DuckDB ingestion.
    """
    if parameters is None:
        parameters = LA_PARAMETERS
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=lookback_days)

    all_readings = []
    for i, station in enumerate(stations):
        sid = station["station_id"]
        for param in parameters:
            try:
                raw = fetch_measurements(sid, param, date_from, date_to)
                for r in raw:
                    ts = (r.get("date") or {}).get("utc", "")
                    all_readings.append({
                        "station_id": sid,
                        "parameter": param,
                        "value": r.get("value"),
                        "unit": r.get("unit", "µg/m³"),
                        "timestamp": ts,
                    })
                print(f"  [{i+1}/{len(stations)}] {station.get('name', sid)} / {param}: {len(raw)} readings")
            except requests.HTTPError as e:
                print(f"  Warning: {param}@{sid} failed — {e}")
    return all_readings


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    Path("data/metadata").mkdir(parents=True, exist_ok=True)

    print("Fetching LA metro stations from OpenAQ v3...")
    raw = fetch_la_stations()
    stations = parse_station_metadata(raw)
    print(f"Found {len(stations)} stations with coordinates.")

    df = pd.DataFrame(stations)
    out = Path("data/metadata/stations.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} stations to {out}")
    print(df[["station_id", "name", "lat", "lon"]].to_string(index=False))
