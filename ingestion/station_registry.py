import json
import math
from pathlib import Path

import pandas as pd
from haversine import haversine

D_CUTOFF_KM = 40.0

# Initial lambda for development. Δelevation is in meters, d_haversine in km.
# Units of lambda: km² / m² — converts elevation² to km²-equivalent.
# Tuned on held-out validation stations in Step 6; expected range 0.0001–0.001
# (100m elevation ≈ 1–3 km horizontal at those values).
LAMBDA_DEFAULT = 0.0005

METADATA_DIR = Path("data/metadata")


def composite_distance(
    s1: dict, s2: dict, lambda_param: float = LAMBDA_DEFAULT
) -> float:
    """
    Spatial distance combining great-circle distance and elevation difference.

        d = sqrt(d_haversine_km² + λ · Δelevation_m²)

    λ units: km²/m². Converts elevation² penalty into km²-equivalent so the
    result is in km. Tuned on held-out stations in Step 6.
    """
    d_h = haversine((s1["lat"], s1["lon"]), (s2["lat"], s2["lon"]))
    d_elev = abs(s1["elevation_m"] - s2["elevation_m"])
    return math.sqrt(d_h**2 + lambda_param * d_elev**2)


def epanechnikov_weight(d: float, d_cutoff: float = D_CUTOFF_KM) -> float:
    """
    Epanechnikov kernel: optimal in MSE sense, reaches exactly zero at d_cutoff.

        w = max(0, 1 - (d / d_cutoff)²)
    """
    if d >= d_cutoff:
        return 0.0
    return max(0.0, 1.0 - (d / d_cutoff) ** 2)


def build_spatial_neighbor_index(
    stations: list[dict],
    lambda_param: float = LAMBDA_DEFAULT,
    d_cutoff: float = D_CUTOFF_KM,
) -> dict[str, list[tuple[str, float]]]:
    """
    Compute kernel-weighted neighbor lists for every station.

    Returns:
        {station_id: [(neighbor_id, normalized_weight), ...]}

    Weights within each station's neighbor list sum to 1.0.
    Stations with no neighbors within d_cutoff map to an empty list.
    """
    index: dict[str, list[tuple[str, float]]] = {}
    for s in stations:
        sid = str(s["station_id"])
        neighbors: list[tuple[str, float]] = []
        for other in stations:
            oid = str(other["station_id"])
            if oid == sid:
                continue
            d = composite_distance(s, other, lambda_param)
            w = epanechnikov_weight(d, d_cutoff)
            if w > 0:
                neighbors.append((oid, w))
        total = sum(w for _, w in neighbors)
        index[sid] = [(nid, w / total) for nid, w in neighbors] if total > 0 else []
    return index


def save_neighbor_index(index: dict, path: Path | None = None) -> None:
    path = path or METADATA_DIR / "neighbor_index.json"
    with open(path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Saved neighbor index to {path}")


def load_neighbor_index(path: Path | None = None) -> dict:
    path = path or METADATA_DIR / "neighbor_index.json"
    with open(path) as f:
        return json.load(f)


def load_stations(path: Path | None = None) -> list[dict]:
    path = path or METADATA_DIR / "station_elevations.csv"
    return pd.read_csv(path).to_dict(orient="records")


def summarize_index(index: dict) -> None:
    """Print neighbor count distribution for visual inspection."""
    counts = [len(v) for v in index.values()]
    if not counts:
        print("No stations in index.")
        return
    print(f"Stations: {len(counts)}")
    print(f"Neighbors per station — min: {min(counts)}, max: {max(counts)}, "
          f"mean: {sum(counts)/len(counts):.1f}")
    isolated = sum(1 for c in counts if c == 0)
    if isolated:
        print(f"Warning: {isolated} station(s) with zero neighbors (consider raising d_cutoff).")


if __name__ == "__main__":
    stations = load_stations()
    print(f"Building spatial neighbor index for {len(stations)} stations "
          f"(λ={LAMBDA_DEFAULT}, d_cutoff={D_CUTOFF_KM}km)...")

    index = build_spatial_neighbor_index(stations)
    summarize_index(index)
    save_neighbor_index(index)

    # Spot-check: print top-3 neighbors for the first station
    first_id = stations[0]["station_id"]
    first_name = stations[0].get("name", first_id)
    print(f"\nTop neighbors for '{first_name}':")
    by_id = {str(s["station_id"]): s.get("name", s["station_id"]) for s in stations}
    for nid, w in sorted(index[str(first_id)], key=lambda x: -x[1])[:5]:
        print(f"  {by_id.get(nid, nid)}: weight={w:.4f}")
