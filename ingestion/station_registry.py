import json
import math
from pathlib import Path

import pandas as pd
from haversine import haversine

D_CUTOFF_KM = 40.0

# λ units: km²/m² — because d_haversine is in km and Δelevation is in meters.
# At λ=0.0005: 100m elevation ≈ 2.2km horizontal, 300m ≈ 6.7km.
# Project plan specifies 0.05–0.20 as a tuning range but that range assumes
# dimensionless (km/km) inputs; in km²/m² units the equivalent range is ~0.00005–0.001.
# Tuned on held-out validation stations in Step 6.
LAMBDA_DEFAULT = 0.0005

METADATA_DIR = Path("data/metadata")

# Stations within this distance are treated as the same physical site.
# Grounded in observed data: same-site GPS precision differences top out at ~161m;
# clear relocations start at ~359m. 250m sits cleanly in the gap.
DEDUP_THRESHOLD_M = 250.0


def deduplicate_stations(
    stations: list[dict],
    threshold_m: float = DEDUP_THRESHOLD_M,
) -> tuple[list[dict], dict[str, str]]:
    """
    Remove co-located duplicate station entries (same physical site, multiple OpenAQ IDs).

    OpenAQ creates new location IDs when instruments are replaced or recalibrated at the
    same physical address. Within 250m, we treat entries as the same site and keep only
    the canonical one (lowest station_id — i.e. the oldest entry).

    Returns:
        canonical: deduplicated station list
        alias_map: {duplicate_station_id: canonical_station_id} for all dropped IDs,
                   so raw_readings from duplicate IDs can still be associated with the
                   canonical station when computing spatial features.
    """
    # Sort ascending by station_id so the earliest (lowest) ID wins as canonical.
    sorted_stations = sorted(stations, key=lambda s: int(s["station_id"]))

    canonical: list[dict] = []
    alias_map: dict[str, str] = {}

    for s in sorted_stations:
        sid = str(s["station_id"])
        merged = False
        for c in canonical:
            d_m = haversine(
                (s["lat"], s["lon"]), (c["lat"], c["lon"])
            ) * 1000
            if d_m <= threshold_m:
                alias_map[sid] = str(c["station_id"])
                merged = True
                break
        if not merged:
            canonical.append(s)

    return canonical, alias_map


def save_alias_map(alias_map: dict, path: Path | None = None) -> None:
    path = path or METADATA_DIR / "station_aliases.json"
    with open(path, "w") as f:
        json.dump(alias_map, f, indent=2)
    print(f"Saved station alias map to {path}")


def load_alias_map(path: Path | None = None) -> dict:
    path = path or METADATA_DIR / "station_aliases.json"
    with open(path) as f:
        return json.load(f)


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
    print(f"Loaded {len(stations)} stations from elevation CSV.")

    canonical, alias_map = deduplicate_stations(stations)
    print(f"After deduplication (threshold={DEDUP_THRESHOLD_M}m): "
          f"{len(canonical)} canonical stations, {len(alias_map)} aliases.")
    if alias_map:
        print("Aliases (duplicate_id → canonical_id):")
        for dup, canon in alias_map.items():
            dup_name = next((s["name"] for s in stations if str(s["station_id"]) == dup), dup)
            print(f"  {dup} ({dup_name}) → {canon}")
    save_alias_map(alias_map)

    print(f"\nBuilding spatial neighbor index for {len(canonical)} canonical stations "
          f"(λ={LAMBDA_DEFAULT}, d_cutoff={D_CUTOFF_KM}km)...")
    index = build_spatial_neighbor_index(canonical)
    summarize_index(index)
    save_neighbor_index(index)

    # Spot-check: top neighbors for the first canonical station
    first = canonical[0]
    first_id = str(first["station_id"])
    print(f"\nTop neighbors for '{first.get('name', first_id)}':")
    by_id = {str(s["station_id"]): s.get("name", s["station_id"]) for s in canonical}
    for nid, w in sorted(index[first_id], key=lambda x: -x[1])[:5]:
        print(f"  {by_id.get(nid, nid)}: weight={w:.4f}")
