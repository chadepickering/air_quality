import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.openaq.org/v3"

# South Coast AQMD bounding box: min_lon, min_lat, max_lon, max_lat (WGS84)
LA_BBOX = "-118.9,33.5,-117.0,34.8"

# Parameters we care about and their OpenAQ v3 IDs
LA_PARAMETERS = ["pm25", "no2", "o3", "pm10", "co"]
PARAM_IDS = {"pm25": 2, "no2": 7, "o3": 9, "pm10": 1, "co": 6}


def _headers() -> dict:
    api_key = os.getenv("OPENAQ_API_KEY")
    return {"X-API-Key": api_key} if api_key else {}


def fetch_la_stations() -> list[dict]:
    """
    Fetch all PM2.5-capable station location objects in the LA metro bbox.
    Each location object includes a sensors[] array used by extract_sensor_index().
    """
    stations = []
    page = 1
    while True:
        params = {
            "bbox": LA_BBOX,
            "parameters_id": PARAM_IDS["pm25"],
            "monitor": "true",   # regulatory reference monitors only; excludes low-cost sensors
            "limit": 100,
            "page": page,
        }
        resp = requests.get(
            f"{BASE_URL}/locations", params=params, headers=_headers(), timeout=30
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        stations.extend(results)
        if len(results) < 100:
            break
        page += 1
        time.sleep(0.5)
    return stations


def parse_station_metadata(raw_stations: list[dict]) -> list[dict]:
    """
    Flatten OpenAQ location objects to station rows for stations.csv.
    Fields: station_id, name, lat, lon, country, timezone.
    """
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


def extract_sensor_index(raw_stations: list[dict]) -> dict[str, dict]:
    """
    Build a sensor_id → {station_id, parameter, unit} index from location objects.

    In OpenAQ v3, measurements are fetched per sensor (not per location+parameter).
    This index maps each sensor ID to the station and parameter it belongs to,
    so fetch_measurements() can route results back to the right station/parameter.

    Returns:
        {
          "67890": {"station_id": "12345", "parameter": "pm25", "unit": "µg/m³"},
          "67891": {"station_id": "12345", "parameter": "no2",  "unit": "ppb"},
          ...
        }
    """
    index = {}
    for loc in raw_stations:
        station_id = str(loc["id"])
        for sensor in loc.get("sensors") or []:
            param = sensor.get("parameter") or {}
            param_name = param.get("name", "")
            if param_name not in LA_PARAMETERS:
                continue
            index[str(sensor["id"])] = {
                "station_id": station_id,
                "parameter": param_name,
                "unit": param.get("units", "µg/m³"),
            }
    return index


def fetch_measurements(
    sensor_id: str,
    date_from: datetime,
    date_to: datetime,
    page_limit: int = 1000,
) -> list[dict]:
    """
    Fetch measurements for one sensor via GET /v3/sensors/{sensor_id}/measurements.
    Returns list of {value, timestamp} dicts; station_id/parameter/unit are in the
    sensor_index and added by fetch_historical_bulk().

    Timestamp is datetimeFrom.utc — the start of the measurement period.
    """
    readings = []
    page = 1
    while True:
        params = {
            "datetime_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "datetime_to": date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": page_limit,
            "page": page,
        }
        resp = requests.get(
            f"{BASE_URL}/sensors/{sensor_id}/measurements",
            params=params,
            headers=_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        for r in results:
            ts = (r.get("datetimeFrom") or {}).get("utc", "")
            readings.append({"value": r.get("value"), "timestamp": ts})
        if len(results) < page_limit:
            break
        page += 1
        time.sleep(0.3)
    return readings


def fetch_historical_bulk(
    sensor_index: dict[str, dict],
    lookback_days: int = 365,
) -> list[dict]:
    """
    Pull historical measurements for every sensor in the index.
    Returns flat list of dicts matching the raw_readings DuckDB schema:
        station_id, parameter, value, unit, timestamp

    quality_flag is not set here — that's Step 4 (sensor validation).
    """
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=lookback_days)

    all_readings = []
    items = list(sensor_index.items())
    for i, (sensor_id, meta) in enumerate(items):
        try:
            raw = fetch_measurements(sensor_id, date_from, date_to)
            for r in raw:
                all_readings.append({
                    "station_id": meta["station_id"],
                    "parameter": meta["parameter"],
                    "value": r["value"],
                    "unit": meta["unit"],
                    "timestamp": r["timestamp"],
                })
            print(
                f"  [{i+1}/{len(items)}] sensor {sensor_id} "
                f"({meta['station_id']} / {meta['parameter']}): {len(raw)} readings"
            )
        except requests.HTTPError as e:
            print(f"  Warning: sensor {sensor_id} ({meta['parameter']}) failed — {e}")
    return all_readings


def save_sensor_index(index: dict, path: Path | None = None) -> None:
    path = path or Path("data/metadata/sensor_index.json")
    with open(path, "w") as f:
        json.dump(index, f, indent=2)


def load_sensor_index(path: Path | None = None) -> dict:
    path = path or Path("data/metadata/sensor_index.json")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import pandas as pd
    from dotenv import load_dotenv
    load_dotenv()

    Path("data/metadata").mkdir(parents=True, exist_ok=True)

    print("Fetching LA metro stations from OpenAQ v3...")
    raw = fetch_la_stations()
    stations = parse_station_metadata(raw)
    sensor_index = extract_sensor_index(raw)

    print(f"Found {len(stations)} stations, {len(sensor_index)} sensors across {LA_PARAMETERS}.")

    pd.DataFrame(stations).to_csv("data/metadata/stations.csv", index=False)
    print("Saved data/metadata/stations.csv")

    save_sensor_index(sensor_index)
    print("Saved data/metadata/sensor_index.json")

    print("\nStation list:")
    pd.DataFrame(stations)[["station_id", "name", "lat", "lon"]].pipe(
        lambda df: print(df.to_string(index=False))
    )
