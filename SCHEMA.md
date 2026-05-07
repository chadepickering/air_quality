# Database and File Schema Reference

This document defines every field in every persistent store produced by this project.
It is the authoritative reference for column semantics — use it when writing queries, models,
or downstream visualizations.

---

## Contents

1. [DuckDB — `raw_readings` table](#1-duckdb--raw_readings-table)
2. [DuckDB — `processed_features` table](#2-duckdb--processed_features-table)
3. [CSV — `data/metadata/stations.csv`](#3-csv--datametadatastationscsv)
4. [JSON — `data/metadata/neighbor_index.json`](#4-json--datametadataneighbor_indexjson)

---

## 1. DuckDB — `raw_readings` table

**File:** `data/processed/aq.duckdb`  
**Primary key:** `(station_id, parameter, timestamp)`  
**Populated by:** `ingestion/aqs_client.py`  
**Row count (5-year pull):** 2,634,473  
**Date range:** 2021-03-01 → 2026-03-01 (UTC)

One row per station × parameter × UTC hour. Timestamps are stored as naive UTC
(timezone info stripped before insert to prevent DuckDB DST conversion artifacts).

| Column | Type | Description |
|---|---|---|
| `station_id` | VARCHAR | AQS site identifier. Format: `"SS-CCC-NNNN"` where SS = 2-digit state FIPS, CCC = 3-digit county FIPS, NNNN = 4-digit site number. Example: `"06-037-0016"` (Los Angeles, CA). Stable across instrument replacements. |
| `parameter` | VARCHAR | Pollutant name. One of: `"pm25"`, `"no2"`, `"o3"`, `"pm10"`, `"co"`. |
| `value` | FLOAT | Raw sensor measurement in native AQS units (see unit column). **Not converted.** Negatives are retained per EPA QA guidance — see quality_flag for interpretation. |
| `unit` | VARCHAR | Unit of measure for this parameter. `"µg/m³"` for pm25 and pm10; `"ppb"` for no2; `"ppm"` for o3 and co. Consistent within each parameter across all rows. |
| `timestamp` | TIMESTAMP | Start of the 1-hour averaging period, stored as naive UTC. AQS `time_gmt` convention: `"08:00"` means the hour 08:00–09:00 UTC. Constructed as `f"{date_gmt}T{time_gmt}:00Z"` with timezone info stripped before insert. |
| `quality_flag` | INTEGER | Data quality flag. See flag definitions below. |
| `ingested_at` | TIMESTAMP | Wall-clock UTC time when the row was written to DuckDB. Set automatically by DuckDB `DEFAULT CURRENT_TIMESTAMP`. Not used by the ML pipeline. |

### quality_flag values

| Value | Label | Set by | Meaning |
|---|---|---|---|
| `0` | Valid | AQS ingestion | AQS returned a blank qualifier for this reading. Value is within expected instrument operating range. |
| `1` | Suspect | AQS ingestion **or** `sensor_validation.apply_validation()` | Either (a) AQS returned a non-blank qualifier string (e.g. `"IH"` for holiday fireworks influence, `"AN"` for machine malfunction, `"Q"` for concurrence required), or (b) the value is outside the suspect bounds defined in `streaming/sensor_validation.py` — most commonly a physically impossible negative concentration. **Suspect rows are retained in `processed_features`; they are not imputed.** |
| `2` | Invalid | `sensor_validation.apply_validation()` | Value exceeds the invalid bounds (see below), indicating instrument malfunction. Invalid rows are treated as missing data and imputed or left NaN in `processed_features`. |

### Why negatives are suspect, not valid

Negative concentrations are physically impossible. However, continuous monitors
(BAM, TEOM-FDMS for PM2.5; UV photometric for O3; chemiluminescence for NO2)
produce small negatives near their detection limit due to temperature/humidity
correction algorithms and electronic noise. EPA QA Handbook Vol. II (Section 2.2.3)
explicitly recommends **retaining** these values rather than zeroing them, because
zeroing introduces upward bias in averages and percentiles. They are flagged
`suspect` (1) — not `invalid` (2) — so the feature engineering pipeline retains
them as measured. Models see the true instrument reading.

Observed 5-year noise floors in the LA metro SCAQMD dataset:

| Parameter | Observed min | Suspect if | Invalid if |
|---|---|---|---|
| PM2.5 | -7.7 µg/m³ | < 0 | < -15.0 |
| NO2 | -1.7 ppb | < 0 | < -10.0 |
| O3 | -0.004 ppm | < 0 | < -0.05 |
| PM10 | -9.0 µg/m³ | < 0 | < -20.0 |
| CO | -0.2 ppm | < 0 | < -1.0 |

Invalid lower bounds are set at approximately 2× the observed noise floor to
ensure legitimate instrument noise is never misclassified as malfunction.

### Parameter reference

| Parameter | AQS code | Unit | Instrument type | 5-yr range (SCAQMD) | EPA standard |
|---|---|---|---|---|---|
| PM2.5 | 88101 | µg/m³ | BAM / TEOM-FDMS | -7.7 to 549 | 35 µg/m³ (24-hr), 9 µg/m³ (annual) |
| NO2 | 42602 | ppb | Chemiluminescence | -1.7 to 95 | 100 ppb (1-hr), 53 ppb (annual) |
| O3 | 44201 | ppm | UV photometric | -0.004 to 0.145 | 0.070 ppm (8-hr) |
| PM10 | 81102 | µg/m³ | BAM / TEOM | -9 to 2276 | 150 µg/m³ (24-hr) |
| CO | 42101 | ppm | NDIR | -0.2 to 10.1 | 9 ppm (8-hr), 35 ppm (1-hr) |

Note: AQS returns native units. The modeling pipeline does **not** convert ppb↔µg/m³.
NO2 lags and spatial features are therefore in ppb; O3 and CO in ppm. Ensure
models are not mixing unit systems when computing composite features.

---

## 2. DuckDB — `processed_features` table

**File:** `data/processed/aq.duckdb`  
**Primary key:** `(station_id, timestamp)`  
**Populated by:** `streaming/feature_engineering.py`  
**Expected row count:** ~833,000 (19 stations × ~43,800 hours per 5-year range)

One row per station × UTC hour. Every hour in each station's observed date range
is present after reindexing — gaps from raw_readings appear here as imputed values
or NaN (for outages > 24 hours).

### Identifiers

| Column | Type | Description |
|---|---|---|
| `station_id` | VARCHAR | AQS site identifier. FK to `raw_readings.station_id`. |
| `timestamp` | TIMESTAMP | UTC hour (naive). Full hourly grid from each station's first to last observed reading. |

### Pollutant measurements (imputed)

These columns hold the best available measurement for each station-hour. Values are:
1. The raw reading (quality_flag 0 or 1), if available.
2. Linearly interpolated, if gap was 1–3 consecutive hours.
3. Same-hour-of-day median from prior 7 days, if gap was 4–24 hours.
4. NaN, if gap exceeded 24 hours (extended outage).

Quality_flag=2 (invalid) readings are treated as missing and enter the imputation chain.
Quality_flag=1 (suspect) readings, including instrument negatives, are used as-is.

| Column | Type | Unit | Description |
|---|---|---|---|
| `pm25` | FLOAT | µg/m³ | PM2.5 concentration. Target variable for forecasting. |
| `no2` | FLOAT | ppb | NO2 concentration. Traffic/combustion covariate. |
| `o3` | FLOAT | ppm | O3 concentration. Photochemical/temperature covariate. |
| `pm10` | FLOAT | µg/m³ | PM10 concentration. Coarse particulate / dust covariate. |
| `co` | FLOAT | ppm | CO concentration. Traffic/combustion covariate. |

### Temporal features

Derived deterministically from the UTC timestamp. No leakage risk.

| Column | Type | Range | Description |
|---|---|---|---|
| `hour_of_day` | INTEGER | 0–23 | UTC hour extracted from timestamp. |
| `day_of_week` | INTEGER | 0–6 | Day of week (Monday=0, Sunday=6). Pandas `dayofweek` convention. |
| `month` | INTEGER | 1–12 | Calendar month. |
| `is_weekend` | BOOLEAN | T/F | True if day_of_week ≥ 5 (Saturday or Sunday). |

### PM2.5 rolling window features

Rolling means computed over the imputed `pm25` series, per station, in chronological
order. `min_periods=1` means the first few rows use partial windows rather than NaN.
These features are computed per-station before any cross-station aggregation.

| Column | Type | Description |
|---|---|---|
| `pm25_roll3` | FLOAT | 3-hour trailing mean of pm25 (µg/m³). |
| `pm25_roll6` | FLOAT | 6-hour trailing mean of pm25 (µg/m³). |
| `pm25_roll24` | FLOAT | 24-hour trailing mean of pm25 (µg/m³). |

### PM2.5 lag features

Lag values reference the same station's own imputed time series. NaN for the first
N rows of each station's history (expected; models learn short-context behavior).

| Column | Type | Description |
|---|---|---|
| `pm25_lag1` | FLOAT | PM2.5 one hour prior at this station (µg/m³). |
| `pm25_lag3` | FLOAT | PM2.5 three hours prior (µg/m³). |
| `pm25_lag24` | FLOAT | PM2.5 twenty-four hours prior (µg/m³). |

### Spatial features

Kernel-weighted aggregates across neighboring stations, using the Epanechnikov
kernel with d_cutoff=40 km and λ=0.0005 km²/m² (see `SCHEMA.md §4` for the
full spatial index definition). Weights are normalized to sum to 1.0 within each
station's neighbor list.

**At each timestamp**, weights are re-normalized over neighbors with non-NaN values —
a neighbor that is offline at time t does not dilute the aggregate with zeros.

Two stations — Lancaster (`06-037-2005`) and Victorville (`06-071-9004`) — have
no neighbors within the 40 km cutoff (Mojave Desert geography). Their spatial
feature columns are NaN for all rows.

| Column | Type | Unit | Description |
|---|---|---|---|
| `spatial_pm25_lag1` | FLOAT | µg/m³ | Kernel-weighted average of neighbor PM2.5 at t−1. Captures the upwind/adjacent pollution field one hour ago. |
| `spatial_pm25_lag3` | FLOAT | µg/m³ | Kernel-weighted average of neighbor PM2.5 at t−3. |
| `spatial_pm25_roll6` | FLOAT | µg/m³ | Kernel-weighted average of neighbor 6-hour rolling PM2.5 mean. Smoothed regional background signal. |
| `spatial_no2_lag1` | FLOAT | ppb | Kernel-weighted average of neighbor NO2 at t−1. Traffic corridor influence. |
| `spatial_o3_lag1` | FLOAT | ppm | Kernel-weighted average of neighbor O3 at t−1. Photochemical regime signal. |
| `spatial_elev_diff` | FLOAT | m | Kernel-weighted absolute elevation difference between this station and each neighbor. Static (time-invariant) — same value for every row of a given station. Represents topographic context: how much higher/lower this site is relative to its neighbors on average. |

### Train/val/test split

| Column | Type | Values | Description |
|---|---|---|---|
| `split` | VARCHAR | `"train"`, `"val"`, `"test"` | Chronological split assignment. Cutoff constants defined in `streaming/feature_engineering.py`. See table below. |

| Split | Period | Approximate duration | Purpose |
|---|---|---|---|
| `train` | 2021-03-01 → 2025-09-30 | ~4.5 years | Model fitting and seasonal pattern learning |
| `val` | 2025-10-01 → 2025-12-31 | 3 months | Hyperparameter tuning, early stopping, spatial λ grid search |
| `test` | 2026-01-01 → end of data | ~2 months | Final held-out evaluation. Never touched until all models are frozen. |

**Leakage rules (enforced in `feature_engineering.py`):**
- Rolling means and lag features use only past data by construction.
- Spatial features look back in time (lag/roll) using only past neighbor readings.
- 7-day median imputation fill values use only data available at each timestamp.
- Z-score scalers must be fit on `split = 'train'` rows only, then applied to val/test.

---

## 3. CSV — `data/metadata/stations.csv`

**Populated by:** `ingestion/aqs_client.py` (`build_station_list()`)  
**Row count:** 19 stations  
**Scope:** Active continuous monitors within the LA metro bounding box
(lat 33.5–34.8, lon -118.9 – -117.0) across five SCAQMD counties as of 2024–2025.

| Column | Type | Description |
|---|---|---|
| `station_id` | string | AQS site identifier (`"SS-CCC-NNNN"`). Primary key. Matches `raw_readings.station_id`. |
| `name` | string | Human-readable site name from AQS `local_site_name` field, falling back to `address` if blank. |
| `lat` | float | Latitude in decimal degrees, WGS84. Positive = North. |
| `lon` | float | Longitude in decimal degrees, WGS84. Negative = West (all LA metro stations are negative). |
| `elevation_m` | float | Station elevation in meters above sea level, sourced directly from the AQS `monitors/byCounty` response. Complete for all 19 stations (no external elevation API needed). |
| `county_code` | string | 3-digit FIPS county code (e.g., `"037"` = Los Angeles, `"059"` = Orange, `"065"` = Riverside, `"071"` = San Bernardino, `"111"` = Ventura). |
| `state_code` | integer | FIPS state code. `6` for California (stored as integer by pandas CSV round-trip). |

### Station coverage note

19 stations are discovered in the bounding box. Of these, 14 operate continuous
hourly PM2.5 instruments (BAM/TEOM-FDMS). The remaining 5 are FRM (filter-based
reference method) sites that collect 24-hour integrated samples — they have no
hourly data in AQS and are therefore absent from `raw_readings` for PM2.5,
though they may appear for other parameters.

---

## 4. JSON — `data/metadata/neighbor_index.json`

**Populated by:** `ingestion/station_registry.py` (`build_spatial_neighbor_index()`)  
**Structure:** `{station_id: [[neighbor_id, weight], ...]}`

Precomputed spatial neighbor relationships used by `streaming/feature_engineering.py`
to construct the spatial feature columns in `processed_features`.

### Spatial distance metric

For each pair of stations (target s, neighbor i):

```
d(s, i) = sqrt(d_haversine(s,i)² + λ × Δelevation(s,i)²)
```

Where:
- `d_haversine` = great-circle distance in **km** (haversine package default)
- `Δelevation` = absolute elevation difference in **meters**
- `λ = 0.0005 km²/m²` (development default; tuned on validation set in Step 6)

At λ=0.0005: a 100m elevation difference adds ~2.2km of effective distance;
300m adds ~6.7km. This penalizes neighbors separated by terrain features
(basin walls, mountain ridges) that disrupt pollutant transport.

### Epanechnikov kernel weighting

```
w_raw(s, i) = max(0, 1 − (d(s,i) / d_cutoff)²)
```

- `d_cutoff = 40 km` (development default; tuned alongside λ in Step 6)
- Stations beyond 40 km receive weight 0 and are excluded from the neighbor list
- Reaches exactly zero at the cutoff — no arbitrary tail truncation

Final weights are normalized within each station's neighbor list to sum to 1.0:

```
w(s, i) = w_raw(s, i) / Σⱼ w_raw(s, j)
```

### Isolated stations

Two stations have no neighbors within 40 km and map to empty lists:

| Station | Name | Reason |
|---|---|---|
| `06-037-2005` | Lancaster | Antelope Valley / Mojave Desert — geographically isolated from basin network |
| `06-071-9004` | Victorville | San Bernardino desert — beyond Cajon Pass from basin network |

Their spatial feature columns in `processed_features` are NaN for all timestamps.
This is expected behavior, not a data pipeline failure.

### JSON structure example

```json
{
  "06-037-0016": [
    ["06-037-1103", 0.4821],
    ["06-037-1302", 0.2934],
    ["06-037-4004", 0.1507],
    ["06-059-0001", 0.0738]
  ],
  "06-037-2005": [],
  ...
}
```

Each inner list is `[neighbor_station_id, normalized_weight]`.
Weights within each station's list sum to 1.0 (verified by `tests/test_spatial_weights.py::test_neighbor_weights_sum_to_one_in_index`).
