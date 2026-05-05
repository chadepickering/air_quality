import time

import requests

EPQS_URL = "https://epqs.nationalmap.gov/v1/json"


def get_elevation(lat: float, lon: float) -> float | None:
    """Query USGS Elevation Point Query Service. Returns elevation in meters or None on failure."""
    params = {"x": lon, "y": lat, "units": "Meters", "includeDate": False}
    try:
        resp = requests.get(EPQS_URL, params=params, timeout=15)
        resp.raise_for_status()
        value = resp.json().get("value")
        if value is None or str(value).strip() in ("-1000000", ""):
            return None
        return float(value)
    except Exception as e:
        print(f"  Warning: USGS elevation failed for ({lat:.4f}, {lon:.4f}): {e}")
        return None


def enrich_stations_with_elevation(stations: list[dict]) -> list[dict]:
    """
    Add elevation_m to each station dict via USGS one-time point queries.
    Stations with failed queries get elevation_m=0.0 (sea level fallback).
    Rate-limited to ~2 requests/second.
    """
    enriched = []
    for i, station in enumerate(stations):
        elev = get_elevation(station["lat"], station["lon"])
        elev_m = elev if elev is not None else 0.0
        enriched.append({**station, "elevation_m": elev_m})
        status = f"{elev_m:.1f}m" if elev is not None else "MISSING (using 0.0m)"
        print(f"  [{i+1}/{len(stations)}] {station.get('name', station['station_id'])}: {status}")
        time.sleep(0.5)
    return enriched


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    stations_path = Path("data/metadata/stations.csv")
    if not stations_path.exists():
        raise FileNotFoundError("data/metadata/stations.csv not found — run ingestion.openaq_client first.")

    stations = pd.read_csv(stations_path).to_dict(orient="records")
    print(f"Enriching {len(stations)} stations with USGS elevation data...")

    enriched = enrich_stations_with_elevation(stations)

    df = pd.DataFrame(enriched)
    out = Path("data/metadata/station_elevations.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved elevation-enriched station metadata to {out}")

    missing = df[df["elevation_m"] == 0.0]
    if not missing.empty:
        print(f"Warning: {len(missing)} stations using fallback elevation 0.0m:")
        print(missing[["station_id", "name"]].to_string(index=False))
