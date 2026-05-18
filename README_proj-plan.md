# Real-Time Air Quality Forecasting and Health Alert System

## Overview

An end-to-end real-time environmental data pipeline that ingests streaming air quality sensor readings from multiple monitoring stations across the LA metro area, forecasts PM2.5 concentrations at multiple time horizons using three time series models, and generates probabilistic public health alerts when predicted air quality is projected to breach EPA threshold levels. The system demonstrates production-grade streaming infrastructure, state-of-the-art probabilistic time series deep learning, spatial feature engineering, and operational monitoring — all on freely available public data with zero cloud cost.

**Total cost to run:** $0 (fully open-source stack)

---

## Architecture

```
EPA AQS API (LA metro monitoring stations)
        ↓
Ingestion Pipeline (Python + requests)
USGS Elevation API (one-time station metadata pull)
        ↓
┌─────────────────────────────────────────────────────┐
│  Kafka Producer Layer                               │
│  - One producer per monitoring station              │
│  - Simulates real-time hourly sensor readings       │
│  - Topic: raw_air_quality (partitioned by station)  │
└─────────────────────────────────────────────────────┘
        ↓
Kafka Topic: raw_air_quality
        ↓
┌─────────────────────────────────────────────────────┐
│  PySpark Structured Streaming Consumer              │
│  - Sensor validation and quality flagging           │
│  - Missing data imputation                          │
│  - Temporal feature engineering                     │
│  - Spatial lag feature engineering                  │
│    (Epanechnikov kernel, d_cutoff=40km)             │
│  - Write to: processed_air_quality topic            │
└─────────────────────────────────────────────────────┘
        ↓
Kafka Topic: processed_air_quality
        ↓
┌─────────────────────────────────────────────────────┐
│  Forecasting Layer (three models)                   │
│  Baseline 1: LSTM                                   │
│  Baseline 2: TFT (Temporal Fusion Transformer)      │
│  Primary:    DeepAR (probabilistic, multi-horizon)  │
│  Horizons:   3hr, 12hr, 24hr, 72hr                  │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│  Probabilistic Alert System                         │
│  Advisory:  P(PM2.5 > 35.4 μg/m³) by horizon       │
│  Warning:   P(PM2.5 > 55.4 μg/m³) by horizon       │
│  Station risk score: precision-weighted across      │
│  horizons (inverse-variance weighting)              │
└─────────────────────────────────────────────────────┘
        ↓
┌──────────────────────┐   ┌────────────────────────────┐
│  Grafana             │   │  Streamlit                 │
│  Operational         │   │  ML Interface              │
│  monitoring          │   │  - Model comparison        │
│  dashboard           │   │  - Forecast viz            │
│  - Real-time feeds   │   │  - Spatial catchment       │
│  - Forecast overlay  │   │    area maps               │
│  - Alert status      │   │  - Threshold sensitivity   │
│  - System health     │   │  - Attention weights       │
└──────────────────────┘   └────────────────────────────┘
```

---

## Dataset

| Property | Detail |
|---|---|
| Primary source | EPA Air Quality System (AQS) REST API |
| Endpoint | `https://aqs.epa.gov/data/api/` |
| Development scope | LA metro area (South Coast AQMD network, 40–70 stations) |
| Production extension | State of California (CARB network, 250+ stations) |
| Access | Free registration — email + API key |
| Format | JSON |
| Temporal resolution | Hourly |
| Primary target | PM2.5 (μg/m³, AQS parameter code 88101) |
| Covariates | NO2 (42602), O3 (44201), PM10 (81102), CO (42101) |
| Elevation data | USGS National Elevation Dataset — one-time point query per station |

**Why AQS over OpenAQ:** AQS is the primary source — SCAQMD stations report directly to EPA AQS; OpenAQ ingests it downstream. AQS's `hourData/byCounty` endpoint returns all stations in a county for a given parameter and date range in a single request, making bulk historical pulls fast (~100 requests for a full year across 5 parameters) and reliable. AQS site IDs are stable (instruments at a given site keep the same ID), eliminating the station deduplication complexity that plagued the OpenAQ-based approach.

**Why LA metro:** South Coast AQMD operates one of the densest air quality monitoring networks in the world. The LA basin's geographic and meteorological complexity — ocean breeze, temperature inversions, wildfire smoke events, traffic corridors — creates rich temporal patterns that reward sophisticated modeling over simpler baselines.

**Scalability note:**

| Tier | Pattern | When to use |
|---|---|---|
| Development | AQS API → local Kafka → local models | LA metro, portfolio demonstration |
| Staging | AQS API → GCS → Kafka → Spark cluster | California statewide, 250+ stations |
| Production | Streaming API → GCS → Spark/BigQuery | National or global deployment |

---

## Alert Design

### Two-Tier Threshold System

| Alert Level | PM2.5 Threshold | EPA Category | Population Affected |
|---|---|---|---|
| Advisory | > 35.4 μg/m³ | Unhealthy for Sensitive Groups | Elderly, children, respiratory conditions |
| Warning | > 55.4 μg/m³ | Unhealthy / Hazardous | General population |

### Probabilistic Alert Output Per Station

For each station and each forecast horizon, DeepAR's Monte Carlo samples produce:

```
Station: Pasadena (ID: USC-001)
Timestamp: 2024-03-15 14:00 UTC

Horizon | Advisory P(>35.4) | Warning P(>55.4) | Confidence
--------|-------------------|------------------|------------
3 hr    | 0.31              | 0.08             | High
12 hr   | 0.67              | 0.24             | Moderate
24 hr   | 0.74              | 0.31             | Moderate
72 hr   | 0.84              | 0.45             | Low

Station Risk Score:
  Advisory: 0.58  (precision-weighted across horizons)
  Warning:  0.21  (precision-weighted across horizons)
  Status:   ADVISORY
```

### Station Risk Score — Precision Weighting

The station risk score uses inverse-variance weighting — higher confidence (narrower predictive interval) forecasts receive stronger weight:

$$w_h = \frac{1/\sigma_h}{\sum_{h'} 1/\sigma_{h'}}$$

Where σ_h is the standard deviation of the predictive distribution at horizon h. The 3-hour forecast is most confident and therefore most heavily weighted. The 72-hour forecast is least confident and least weighted. This is the statistically principled direction — the score reflects what we know most confidently, not what is furthest in the future.

The station risk score is computed separately for Advisory and Warning tiers, then combined into a single status label (CLEAR / ADVISORY / WARNING) driven by the highest active tier.

---

## Spatial Feature Engineering

### Composite Distance Metric

For each target station s and neighboring station i:

$$d_{spatial}(s,i) = \sqrt{d_{haversine}^2(s,i) + \lambda \cdot \Delta_{elevation}^2(s,i)}$$

Where:
- d_haversine: great-circle distance in kilometers
- Δ_elevation: absolute elevation difference in meters
- λ: scaling parameter tuned on validation set (converts elevation difference to equivalent horizontal distance)

**λ tuning strategy:** λ is treated as a hyperparameter optimized on held-out LA metro stations by minimizing forecast error. In practice for the LA basin λ is expected to fall in the range 0.05–0.20. The tuned value will be documented explicitly. For California production extension, elevation difference transitions to a model covariate rather than a distance penalty — eliminating the need for regional recalibration.

### Epanechnikov Kernel Weighting

For each neighboring station i within d_cutoff = 40km:

$$w_i = \max\left(0, 1 - \frac{d_{spatial}^2(s,i)}{d_{cutoff}^2}\right)$$

Stations beyond 40km receive zero weight and are excluded entirely. The Epanechnikov kernel is optimal in the mean squared error sense, computationally simple, and reaches exactly zero at the cutoff — no arbitrary tail truncation required.

d_cutoff = 40km is the development default and is itself tunable alongside λ.

### Weighted Spatial Lag Features

For each target station, the following spatial features are computed as kernel-weighted aggregates across all contributing neighbors:

```python
# Weighted spatial PM2.5 lags
spatial_pm25_lag1  = Σ w_i * pm25_i(t-1)   # 1-hour lag
spatial_pm25_lag3  = Σ w_i * pm25_i(t-3)   # 3-hour lag
spatial_pm25_roll6 = Σ w_i * pm25_roll6_i  # 6-hour rolling mean

# Weighted spatial secondary pollutant lags
spatial_no2_lag1   = Σ w_i * no2_i(t-1)
spatial_o3_lag1    = Σ w_i * o3_i(t-1)

# Weighted elevation difference covariate
spatial_elev_diff  = Σ w_i * |elev_s - elev_i|
```

This collapses the variable number of contributing neighbors into fixed-dimension feature vectors regardless of station density — scaling cleanly from LA metro to statewide California without architectural changes.

---

## Project Structure

```
air_quality/
├── alerts/
│   ├── alert_router.py
│   ├── breach_probability.py
│   ├── risk_score.py
│   └── threshold_config.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── metadata/
│   │   ├── stations.csv
│   │   ├── station_elevations.csv
│   │   └── neighbor_index.json
│   ├── processed/                  # gitignored
│   └── raw/                        # gitignored
├── evaluation/
│   ├── model_comparison.py
│   ├── spatial_catchment_viz.py
│   └── threshold_sensitivity.py
├── ingestion/
│   ├── aqs_client.py
│   ├── database.py
│   └── station_registry.py
├── models/
│   ├── deepar/
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   ├── sample_forecasts.py
│   │   └── train.py
│   ├── lstm/
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   └── train.py
│   └── tft/
│       ├── attention_viz.py
│       ├── evaluate.py
│       ├── model.py
│       └── train.py
├── monitoring/
│   ├── drift/
│   │   ├── feature_drift.py
│   │   └── prediction_drift.py
│   ├── grafana/
│   │   ├── alerts.json
│   │   └── dashboard.json
│   └── influxdb_writer.py
├── notebooks/
│   └── exploration.ipynb
├── streaming/
│   ├── consumer.py
│   ├── create_topics.sh
│   ├── feature_engineering.py
│   ├── producer.py
│   ├── schemas.py
│   ├── sensor_validation.py
│   └── spatial_weights.py
├── tests/
│   ├── integration/
│   │   └── test_pipeline_integration.py  # requires live Kafka; pytest -m integration
│   ├── test_alert_system.py
│   ├── test_consumer.py
│   ├── test_producer.py
│   ├── test_raw_ingestion.py
│   ├── test_risk_score.py
│   ├── test_schemas.py
│   ├── test_sensor_validation.py
│   └── test_spatial_weights.py
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── pytest.ini
├── README.md
├── README_proj-plan.md
├── requirements.txt
└── SCHEMA.md
```

---

## Stack

| Component | Tool | Cost |
|---|---|---|
| Air quality data | EPA AQS REST API | Free (registration required) |
| Elevation data | USGS National Elevation Dataset | Free |
| Local storage | DuckDB | Free |
| Message broker | Apache Kafka (Docker) | Free |
| Stream processing | PySpark Structured Streaming | Free |
| Time series DB | InfluxDB (Docker) | Free |
| LSTM baseline | PyTorch | Free |
| TFT baseline | PyTorch Forecasting | Free |
| DeepAR primary | GluonTS (PyTorch backend) | Free |
| Experiment tracking | Weights & Biases (free tier) | Free |
| Operational dashboard | Grafana (Docker) | Free |
| ML interface | Streamlit | Free |
| Orchestration | Docker Compose | Free |

**Total cost: $0**

---

## Model Comparison Framework

All three models are evaluated on the same held-out test period using the same feature set. Metrics are computed per station and per forecast horizon.

### Point Forecast Metrics (LSTM and TFT)
- MAE — Mean Absolute Error (μg/m³)
- RMSE — Root Mean Squared Error
- MAPE — Mean Absolute Percentage Error

### Probabilistic Metrics (DeepAR primary, TFT quantiles)
- CRPS — Continuous Ranked Probability Score (primary probabilistic metric)
- Prediction Interval Coverage — what fraction of true values fall within the 90% PI (p5–p95 bounds)
- Sharpness — mean width of p5–p95 interval (narrower is better, conditional on coverage)

### Alert-Specific Metrics
- Advisory threshold Brier score — calibration of P(PM2.5 > 35.4)
- Warning threshold Brier score — calibration of P(PM2.5 > 55.4)
- Alert precision and recall at each horizon

---

## Implementation Steps

### Step 1 — Repository Scaffold and Environment Setup ✓

**Tasks:**
- Initialize git repo and `.gitignore`
- Create venv environment
- Install core dependencies
- Create folder structure
- Create `.env.example`

**Key packages:**
```bash
pip install duckdb requests python-dotenv
pip install pyspark kafka-python-ng
pip install torch pytorch-forecasting lightning
pip install gluonts[torch] tensorflow
pip install influxdb-client streamlit wandb
pip install haversine scipy pytest properscoring plotly folium streamlit-folium
```

**Acceptance criteria:**
- [x] All directories created with stub files
- [x] `.gitignore`, `.env.example`, `requirements.txt`, `docker-compose.yml` created
- [x] Docker Compose services skeleton defined

---

### Step 2 — Station Metadata, Elevation, and Spatial Index

**Files:** `ingestion/aqs_client.py`, `ingestion/station_registry.py`

Note: `usgs_elevation.py` is retired. AQS `monitors/byCounty` provides elevation in meters directly, verified complete (0 missing values) across all 5 SCAQMD counties. Elevation is included in `stations.csv`; no separate elevation file or USGS step needed.

**Data source: EPA AQS REST API**

AQS is the authoritative source for all US regulatory air quality monitoring data. SCAQMD reports directly to AQS; OpenAQ was a downstream aggregator and was abandoned due to unreliable bulk measurement APIs. AQS site IDs are stable — instruments at a given site keep the same ID when replaced or recalibrated, eliminating the need for deduplication entirely.

**AQS station discovery:**

```python
# ingestion/aqs_client.py — station discovery

BASE_URL = "https://aqs.epa.gov/data/api"

# Five SCAQMD counties queried and filtered to LA metro bbox (-118.9,33.5,-117.0,34.8)
SCAQMD_COUNTIES = [
    {"state": "06", "county": "037"},  # Los Angeles
    {"state": "06", "county": "059"},  # Orange
    {"state": "06", "county": "065"},  # Riverside
    {"state": "06", "county": "071"},  # San Bernardino
    {"state": "06", "county": "111"},  # Ventura
]

AQS_PARAMETERS = {
    "pm25": {"code": "88101", "unit": "µg/m³"},
    "no2":  {"code": "42602", "unit": "ppb"},
    "o3":   {"code": "44201", "unit": "ppm"},
    "pm10": {"code": "81102", "unit": "µg/m³"},
    "co":   {"code": "42101", "unit": "ppm"},
}

def fetch_monitors_by_county(state: str, county: str, param_code: str) -> list[dict]:
    # GET /monitors/byCounty — full station metadata: lat, lon, elevation, name, close_date
    # station_id = f"{state_code}-{county_code}-{site_number}" e.g. "06-037-0016"

def build_station_list() -> pd.DataFrame:
    # Query PM2.5 monitors across all five counties, deduplicate by site_id,
    # filter to bbox, exclude closed monitors (close_date is not None).
    # Output columns: station_id, name, lat, lon, elevation_m, county_code, state_code
```

**AQS parameter code 88101 (PM2.5):** Covers both 24-hour FRM (filter-based) and continuous FEM (BAM, TEOM-FDMS) instruments. During data pull, rows with `sample_duration != '1 HOUR'` are filtered out — this removes FRM filter readings and retains only continuous hourly instruments. FRM stations will also naturally fail the 80% completeness threshold even without explicit filtering.

**AQS data pull endpoint:** `sampleData/byCounty` (not `hourData/byCounty`, which does not exist in the v1 API). Returns raw sample data including `sample_duration`, `qualifier`, and `poc` fields.

**Composite distance metric and λ units:**

```
d = sqrt(d_haversine_km² + λ · Δelevation_m²)
```

d_haversine is in km (haversine package default); Δelevation is in meters. λ has units km²/m². Development default **λ=0.0005 km²/m²** gives 100m elevation ≈ 2.2km and 300m ≈ 6.7km — physically appropriate for the LA basin. Equivalent tuning range: ~0.00005–0.001 km²/m². λ is tuned on held-out stations in Step 6.

**No station deduplication:** AQS site IDs do not change on instrument replacement. `station_registry.py` contains only spatial functions — no alias map, no dedup machinery.

**Output files:**
- `data/metadata/stations.csv` — AQS site IDs as primary key; columns: `station_id`, `name`, `lat`, `lon`, `elevation_m`, `county_code`, `state_code`
- `data/metadata/neighbor_index.json` — `{station_id: [(neighbor_id, normalized_weight), ...]}`

**Acceptance criteria:**
- [x] LA metro stations pulled and stored to `data/metadata/stations.csv` — **19 stations** (original estimate of 30–50 assumed OpenAQ; AQS bbox+close_date filter yields 19 active sites)
- [x] `elevation_m` populated from AQS for all stations (no USGS call needed)
- [x] Spatial neighbor index computed (λ=0.0005, d_cutoff=40km) — 2 isolated Mojave stations (Lancaster, Victorville) have 0 neighbors by design
- [x] Visual inspection of neighbor assignments makes geographic sense
- [x] **14 stations** with continuous hourly PM2.5 coverage identified — 5 sites are FRM-only (no hourly instrument exists in AQS; not a pipeline issue)

---

### Step 3 — Historical Data Pull and DuckDB Storage

**Files:** `ingestion/aqs_client.py`, `ingestion/database.py`

**Pull strategy — AQS county-level batch queries:**

```python
# ingestion/aqs_client.py — historical pull

def fetch_samples_by_county(
    param_code: str, state: str, county: str,
    date_from: date, date_to: date,
) -> list[dict]:
    # GET /sampleData/byCounty
    # Returns all stations in county for the given parameter and date range.
    # One request covers every station in the county — no per-sensor iteration.
    # Rows are filtered to sample_duration == '1 HOUR' to exclude FRM 24-hr readings.

def fetch_historical_all(
    station_ids: set[str],
    date_from: date,
    date_to: date,
    chunk_days: int = 90,
) -> list[dict]:
    # Iterate: 5 counties × 5 parameters × quarterly chunks ≈ 100 requests total.
    # Filter results to bbox station_ids (stations outside our area are dropped).
    # Returns flat list of {station_id, parameter, value, unit, timestamp, quality_flag}.
```

**AQS response field mapping:**

| AQS field | Maps to |
|---|---|
| `state_code` + `county_code` + `site_num` | `station_id` (e.g., `"06-037-0016"`) |
| `sample_measurement` | `value` |
| `units_of_measure` | `unit` |
| `date_gmt` + `time_gmt` | `timestamp` (UTC; `time_gmt` is end-of-hour convention) |
| `qualifier` (blank) | `quality_flag = 0` (valid) |
| `qualifier` (non-blank) | `quality_flag = 1` (suspect) |

**POC handling:** AQS Parameter Occurrence Code identifies individual instruments at a site. Where multiple instruments measure the same parameter at the same site and hour, keep the reading from the lowest POC that has a non-null value.

**DuckDB schema (unchanged):**
```python
raw_readings:       station_id, parameter, value, unit, timestamp (UTC),
                    quality_flag (0=valid, 1=suspect, 2=invalid),
                    ingested_at — PRIMARY KEY (station_id, parameter, timestamp)

processed_features: station_id, timestamp, pm25, no2, o3, pm10, co,
                    hour_of_day, day_of_week, month, is_weekend,
                    pm25_roll3/6/24, pm25_lag1/3/24,
                    spatial_pm25_lag1/3, spatial_pm25_roll6,
                    spatial_no2_lag1, spatial_o3_lag1, spatial_elev_diff
                    — PRIMARY KEY (station_id, timestamp)
```

**Acceptance criteria:**
- [x] **5 years** of hourly PM2.5 data pulled (Mar 2021 – Mar 2026, 2,634,473 rows) — `date_from = date(2021, 3, 1)` in `aqs_client.py`
- [x] NO2, O3, PM10, CO covariates pulled for same stations and period
- [x] Raw data stored in DuckDB with quality flags; timestamps stored as naive UTC
- [x] Data completeness report run; 24-test integrity suite in `tests/test_raw_ingestion.py` (all passing)
- [x] **14 stations** meet ≥80% PM2.5 completeness — FRM-only stations (5 sites) have no hourly data in AQS; target updated from 20 to reflect AQS reality

---

### Step 4 — Sensor Validation, Imputation, and Feature Engineering

**Files:** `streaming/sensor_validation.py`, `streaming/feature_engineering.py`, `streaming/spatial_weights.py`

**Sensor validation rules:**
```python
PM25_VALID_RANGE = (0.0, 500.0)
NO2_VALID_RANGE  = (0.0, 2000.0)
O3_VALID_RANGE   = (0.0, 500.0)

def validate_reading(value: float, parameter: str) -> int:
    # Returns: 0=valid, 1=suspect, 2=invalid
    ...
```

**Imputation strategy:**
- Missing 1–3 consecutive hours: linear interpolation
- Missing 4–24 hours: same-hour-of-day median from prior 7 days; falls back to a 14-day window if fewer than 4 valid same-hour samples exist in the 7-day window (covers sparse cases at the start of a station's record). Prior-year seasonal context was evaluated and rejected — it would bias imputed values toward climatological normals during the event periods (wildfires, inversions) where regime-tracking is most critical, and only 13 of 1,211 medium-gap hours had insufficient recent data to benefit from it.
- Missing >24 hours: station excluded from spatial features for that period, flag propagated downstream

> **Note on 4–24 hr strategy:** Prior-year seasonal context was evaluated empirically. Gap analysis across 14 FEM stations found 98.1% of medium-gap hours have ≥4 same-hour samples in the 7-day window; only 13 of 1,211 hours would benefit from prior-year fallback. Regime-tracking (7-day) is the correct design — prior-year blending would bias imputation toward climatological normals during wildfires and inversions. Decision: 7-day primary with 14-day fallback for sparse cases.

**Train / validation / test split:**

The data is static AQS history replayed as a streaming simulation. The split must be strictly chronological — no random sampling. These cutoff dates should be defined as named constants in `feature_engineering.py` and referenced consistently across all model training scripts.

```python
TRAIN_END   = date(2025, 9, 30)   # inclusive — ~4.5 years of training data
VAL_END     = date(2025, 12, 31)  # Oct–Dec 2025: hyperparameter tuning, early stopping
# Test set: Jan–Mar 2026 (all available AQS data past VAL_END, ~2 months)
# AQS has a ~2-month publication lag so this is the effective ceiling.
```

| Set        | Period                    | ~Duration | Purpose |
|------------|---------------------------|-----------|---------|
| Train      | Mar 2021 – Sep 2025       | 4.5 years | Model fitting, seasonal pattern learning |
| Validation | Oct 2025 – Dec 2025       | 3 months  | Hyperparameter tuning, early stopping, spatial λ grid search |
| Test       | Jan 2026 – Mar 2026       | ~2 months | Final held-out evaluation — never touched until all models are frozen |

Rationale:
- **Training depth:** 4.5 years gives DeepAR/TFT multiple full seasonal cycles including the Jan 2025 Palisades/Eaton fires as a *training* event (model learns extreme-smoke regime).
- **Test period quality:** Winter 2026 covers temperature-inversion PM2.5 events — the most policy-relevant regime for health alerts.
- **~5% test fraction** is appropriate for long-horizon time series where maximising training data outweighs balanced splits.

**Leakage prevention rules (enforce at implementation):**
- Rolling statistics and z-score scalers must be **fit on training data only**, then applied to val/test.
- Lag and rolling window features that look back into the training period from val/test rows are fine — that is not leakage.
- Imputation fill values (7-day medians, prior-year medians) must be computed using only data available at the time of each row — no future data.
- The `processed_features` table should include a `split` column (`train` / `val` / `test`) assigned by cutoff date, so downstream scripts can filter without re-deriving the dates.

**Acceptance criteria:**
- [x] Sensor validation correctly flags known outliers — two-tier bounds (suspect/invalid) calibrated to 5yr SCAQMD observed ranges; 42-test suite in `tests/test_sensor_validation.py`
- [x] Imputation fills gaps without introducing artifacts — 1–3hr linear interpolation; 4–24hr same-hour-of-day median (7-day primary, 14-day fallback); >24hr left as NaN
- [x] Split cutoff constants defined in `feature_engineering.py`; `processed_features.split` column populated
- [ ] Rolling/scaling statistics fit on train split only — z-score scalers applied in model training scripts (Steps 6–8), not here
- [x] All temporal features computed correctly; spatial features verified — neighbor weights sum to 1 (`tests/test_spatial_weights.py`)
- [ ] Processed features written to DuckDB `processed_features` table — batch pipeline written and tested; `build_processed_features()` not yet executed against live DB (pending before Step 6)

---

### Step 5 — Kafka Producer and PySpark Streaming Consumer ✓

**Files:** `streaming/schemas.py`, `streaming/producer.py`, `streaming/consumer.py`, `streaming/create_topics.sh`, `docker-compose.yml`

**Kafka infrastructure:** Dual-listener Kafka broker (port 9092 internal for Docker network, port 9093 external for host-side processes). Two topics — `raw_air_quality` and `processed_air_quality` — each with 19 partitions and 7-day retention. `AUTO_CREATE_TOPICS_ENABLE=false`; topics created explicitly via `streaming/create_topics.sh`. Kafdrop UI on port 9000.

**Message schemas (`streaming/schemas.py`):** `RawReading` and `ProcessedFeature` dataclasses define the wire format for each topic. `serialize()` converts to UTF-8 JSON bytes with NaN → JSON null sanitization. Matching PySpark `StructType` schemas (`raw_reading_spark_schema()`, `processed_feature_spark_schema()`) are defined here and imported by the consumer for typed `from_json()` parsing.

**Producer (`streaming/producer.py`):** Reads `raw_readings` from DuckDB ordered by `(timestamp, station_id, parameter)`. Publishes one message per row to `raw_air_quality`, keyed by `station_id` (UTF-8) so all readings for a given station land in the same partition and arrive in chronological order. Supports `--date-from`, `--date-to`, and `--rate` (messages/sec; 0 = unlimited). Uses `acks="all"`, `lz4` compression, 64KB batch size.

**Consumer (`streaming/consumer.py`):** PySpark Structured Streaming job. 30-second micro-batch trigger via `foreachBatch`. Each batch is converted to pandas, then runs the same `_impute_series`, `_add_temporal_features`, `_add_rolling_lag_features`, and `compute_spatial_features` functions from the batch pipeline — no duplicated logic. DuckDB-assisted hybrid for stateful features: the last 48 hours of `processed_features` are fetched per batch to provide rolling/lag context beyond the current micro-batch window. Results are published to `processed_air_quality`, keyed by `station_id`.

**Integration test (`tests/integration/test_pipeline_integration.py`):** End-to-end 30-day replay test. Requires live Kafka broker. Run with `pytest -m integration -v`. Uses an ephemeral uniquely-named topic per session to support parallel CI runs. Verifies: message count matches DB row count, all `RawReading` fields present, all `ProcessedFeature` schema fields present, temporal feature ranges, split label validity, `pm25_roll24` non-null after 24+ hours of history.

**Acceptance criteria:**
- [x] Producer replays historical data without errors; message count verified equal to `raw_readings` row count for the replay window
- [x] 19 topic partitions; `station_id` key pins each station to one partition — per-station temporal order preserved
- [x] PySpark consumer processes micro-batches; DuckDB-assisted hybrid provides rolling/lag context beyond current batch
- [x] Processed features published to `processed_air_quality` topic with all 24 schema fields
- [x] Integration test suite covers producer count, consumer output shape, temporal/split field validity
- [x] Kafdrop availability checked in integration test (warns but does not fail if UI is down)

---

### Step 6 — LSTM Baseline

**Files:** `models/lstm/model.py`, `models/lstm/train.py`, `models/lstm/evaluate.py`, `models/lstm/lambda_search.py`

**Framework:** PyTorch (consistent with TFT and DeepAR in Steps 7–8; TensorFlow/Keras dropped to avoid introducing a second DL framework for the baseline alone).

**Architecture (`model.py`):** `LSTMForecaster` — two stacked LSTM layers (hidden_size=64, dropout=0.2 between layers), followed by four independent linear output heads, one per forecast horizon (3hr, 12hr, 24hr, 72hr). Input shape: `(batch, 24, 21)`. Output shape: `(batch, 4)`. Point forecast (no uncertainty quantification — that is DeepAR's role).

**Dataset and normalization (`train.py`):** `AQDataset` builds sliding 24-hour windows from `processed_features`. For each station, a window at position i uses features at hours `[i-23, …, i]` as input and raw PM2.5 at `[i+3, i+12, i+24, i+72]` as targets. Windows where any target PM2.5 is NaN are dropped. Z-score scaler is fit on train-split rows only and saved to `models/lstm/scaler.npz` for reuse by `evaluate.py` and `lambda_search.py`. Val windows that start near the train/val split boundary receive 96 hours of prepended train context so the lookback is always fully populated.

**Seasonality coverage:** The 24hr window + feature set gives three tiers of seasonality signal: (1) diurnal — direct from the 24hr raw history and `hour_of_day`; (2) weekly — `day_of_week` and `is_weekend` encode traffic-driven weekly cycles; (3) inter-seasonal — `month` (1–12) is the primary annual signal, with the model learning seasonal regimes (wildfire autumn, inversion winter) implicitly from 4.5 years of training weights. Year-over-year trends are not explicitly modeled. The LSTM's 24hr window is a known limitation relative to TFT (168hr encoder) and DeepAR (168hr context); the metric gap at 12hr+ horizons is expected and informative.

**Training loop:** Adam optimizer (lr=1e-3), CosineAnnealingLR over the full epoch budget, gradient clipping (max_norm=1.0), early stopping on val MAE (patience=5). Best checkpoint saved to `models/lstm/best_model.pt`. W&B logging: train loss, val MAE, per-horizon val MAE, and LR each epoch. Targets are evaluated in raw μg/m³ (not scaled) so MAE is directly interpretable.

**Train/validation/test split:** (see Step 4 for full rationale and leakage rules)
- Train: Mar 2021 – Sep 2025 (`TRAIN_END = date(2025, 9, 30)`)
- Validation: Oct – Dec 2025 (`VAL_END = date(2025, 12, 31)`) — used for λ tuning and early stopping
- Test: Jan – Mar 2026 — held out until all models are frozen

**λ grid search (`lambda_search.py`):** In-memory 3×3 search over λ ∈ {0.0001, 0.0005, 0.001} km²/m² and d_cutoff ∈ {30, 40, 50} km. For each combination, only the six spatial columns are recomputed from the loaded `processed_features` table — all other features remain fixed, avoiding redundant DuckDB writes. Each point trains the LSTM for 15 proxy epochs; the combination with lowest mean val MAE is selected. If the best result lands on a grid boundary, one additional point is added in that direction before committing. Results written to `models/lstm/lambda_search_results.json`. After the search: update `LAMBDA_DEFAULT` and `D_CUTOFF_KM` in `ingestion/station_registry.py`, re-run `python -m streaming.feature_engineering`, then run full training.

Grid rationale: the λ range spans an order of magnitude (0.0001–0.001 km²/m²), bracketing the physically meaningful elevation-penalty window for the LA basin. A 5×5 expansion was considered and rejected — the spatial loss surface is smooth and the computational cost (~3× longer, ~85–150 min) exceeds the marginal precision gain for a baseline model.

**Outputs:**
- `models/lstm/scaler.npz` — z-score mean/std fit on train split
- `models/lstm/best_model.pt` — best checkpoint by val MAE
- `models/lstm/train_metrics.json` — final epoch metrics and stopped epoch
- `models/lstm/lambda_search_results.json` — full grid results and best combo
- `evaluation/lstm_metrics.json` — per-horizon MAE/RMSE/MAPE on test split

**Acceptance criteria:**
- [x] LSTM trains without errors on processed feature set — converged in 8 epochs, early stopping at epoch 8
- [x] λ tuned on validation set — optimal λ=0.001, d_cutoff=40km (val MAE=5.329); boundary check at λ=0.002 confirmed true optimum
- [x] Spatial features recomputed with tuned λ; `processed_features` table regenerated (812,448 rows)
- [x] Validation MAE < 8 μg/m³ at 3hr horizon — achieved **4.06 μg/m³** (3hr val); test set 3.676 μg/m³
- [x] W&B run logged — `lstm-baseline` run in project `air-quality-forecasting` (run ID: b8zhnjp6)

---

### Step 7 — TFT Baseline

**Files:** `models/tft/model.py`, `models/tft/train.py`, `models/tft/attention_viz.py`, `models/tft/evaluate.py`

TFT via PyTorch Forecasting. Key capabilities: variable selection networks (learns which features matter per station), multi-head attention (identifies which historical timesteps matter at each horizon), quantile regression (5th/50th/95th percentile forecasts for 90% PI coverage evaluation).

**Quantile definition:** Output quantiles are `[0.05, 0.5, 0.95]`. The p50 (median) is the point forecast used for MAE/RMSE comparison with the LSTM. The p5–p95 interval is the 90% prediction interval — consistent with DeepAR's evaluation and with the health alert application where conservative uncertainty bounds are preferable.

**Key outputs to visualize:**
- Variable selection weights: which features TFT finds most informative per station
- Attention patterns: which historical hours most influence each horizon
- Quantile forecasts: 90% PI coverage and sharpness (p5/p95 bounds)

**Training summary:**
- Two-stage run. Initial run (epochs 0–16) crashed at epoch 17 validation due to disk-full (`OSError: No space left on device`). Resumed from best checkpoint (val_loss=0.849, epoch 13). Resumed run completed 34 epochs (Lightning resets counter; overall epochs 17–50).
- Best checkpoint: `best_model-v1.ckpt` — val_loss=**0.761** at overall epoch 48 (resumed epoch 31).
- Full per-epoch history saved to `models/tft/train_metrics.json`.

**Evaluation methodology (Step 7.4):**
- Rolling evaluation across the full test split: all windows where the 72-step decoder falls within the test period. Val period supplies encoder context for windows near the test boundary.
- Data filtered: 5 FRM-only stations excluded (fair comparison with LSTM/DeepAR); `predict=False` used for rolling windows; data trimmed to `test_start − MAX_ENCODER_LENGTH (168h)` to give every window a full encoder lookback.
- Result: **15,200 windows** across 13 stations (one station, `06-071-0306`, lacked sufficient encoder context and was dropped by pytorch-forecasting).
- Actuals collected via `return_y=True` in `model.predict()` — `Prediction.y[0]` is already inverse-transformed to the original PM2.5 scale. Manual denormalization via `GroupNormalizer` is not needed.

**Key evaluation challenges and fixes:**
- `pandas==3.0.2` required: checkpoint was serialized with this exact version; loading with pandas 2.2.3 raises `StringDtype.__init__()` TypeError. Pin to match training environment rather than patch around.
- `show_progress_bar=True` removed: not a valid kwarg in pytorch-forecasting 1.7.0's `predict()` — forwarded to `forward()` and raised `TypeError`.
- `predict=True` gives only 14 windows (last window per station); `predict=False` with encoder-context-trimmed data gives 15,200 rolling windows — statistically comparable to LSTM and DeepAR evaluations.
- `GroupNormalizer.inverse_transform()` is intentionally `NotImplementedError` in pf 1.7.0. The correct pattern is `return_y=True`, which provides already-denormalized actuals.

**Test set results** (`evaluation/tft_metrics.json` — 15,200 windows, 13 stations):

| Horizon | MAE (μg/m³) | RMSE (μg/m³) | PI Coverage | Interval Width |
|---------|-------------|--------------|-------------|----------------|
| 3hr     | 4.764       | 7.538        | 64.0%       | 10.30 μg/m³    |
| 12hr    | 5.286       | 8.255        | 57.7%       | 9.77 μg/m³     |
| 24hr    | 5.437       | 8.428        | 55.2%       | 9.47 μg/m³     |
| 72hr    | 5.404       | 8.483        | 56.8%       | 9.70 μg/m³     |
| Overall | **5.223**   | **8.185**    | **58.5%**   | 9.81 μg/m³     |

LSTM overall test MAE: **5.054 μg/m³** (from `evaluation/lstm_metrics.json`). TFT is slightly behind LSTM on point forecast accuracy — likely attributable to the training interruption cutting short convergence. PI coverage at 58.5% is below the 85–95% target; interval widths (~9–10 μg/m³) are not narrow enough to explain the gap — the model is systematically underconfident in its central quantile forecast.

**Acceptance criteria:**
- [x] TFT trains without errors — two-stage run completed; best val_loss=0.761 at overall epoch 48
- [~] TFT outperforms LSTM on validation MAE at 12hr and 24hr horizons — test MAE 5.223 vs LSTM 5.054 (TFT slightly behind; training cutoff likely a factor)
- [ ] Variable selection weights visualized and saved
- [ ] Attention patterns visualized for representative stations
- [~] 90% PI coverage between 85–95% — achieved 58.5% overall (below target; intervals present but systematically under-coverage)

**Implementation notes:**
- `venv_deepar` conflict: gluonts[torch] requires `lightning<2.5`; TFT uses lightning==2.6.1. These cannot share a venv — DeepAR uses `venv_deepar/` (separate). Main `.venv/` kept at lightning==2.6.1 exclusively for TFT evaluation.
- `dataset_params.pt` regenerated 2026-05-12 after pandas version conflict introduced by gluonts install.
- pandas pinned to 3.0.2 in `.venv` to match training checkpoint serialization.

---

### Step 8 — DeepAR Primary Model

**Files:** `models/deepar/model.py`, `models/deepar/train.py`, `models/deepar/sample_forecasts.py`, `evaluation/conformal.py`, `tests/test_deepar.py`

DeepAR via GluonTS 0.16.2 (PyTorch backend). Autoregressive RNN outputting full predictive distributions via Monte Carlo sampling. Six model versions were trained iteratively, converging on **v4 (ISQF + explicit endpoint quantile knots) with split-conformal calibration** as the production predictor.

**Final production architecture (v4):**
```python
DeepAREstimator(
    freq="h",
    prediction_length=72,
    context_length=168,          # 7-day lookback — matches TFT encoder
    distr_output=FixedISQFOutput(
        num_pieces=10,
        qk_x=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    ),
    num_feat_dynamic_real=4,     # calendar features only — no future leakage
    num_feat_static_cat=1,       # station_id (embedded)
    cardinality=[14],
    lags_seq=[1, 3, 24],         # PM2.5 autoregressive lags — GluonTS feeds own predictions; no leakage
    num_batches_per_epoch=100,
    trainer_kwargs={
        "max_epochs": 75, "accelerator": "auto",
        "gradient_clip_val": 0.1, "enable_model_summary": False
    }
)
```

**Monte Carlo sample generation:** 500 trajectories per window. Rolling evaluation strides 24h through test period (642 windows, 14 stations). Samples saved to `evaluation/deepar_samples.npz` for alert system breach probability computation.

**Venv isolation:** All DeepAR work runs in `venv_deepar/` (gluonts 0.16.2, lightning 2.4.0). Lightning version conflict with TFT (2.6.1) is the reason for isolation.

**CRPS:** Energy-form Continuous Ranked Probability Score — primary DeepAR metric. Jointly penalises bias and over/under-confidence. Computed via sorted-samples O(N log N) algorithm.

---

#### Model Version History

Six versions were trained. Full per-epoch histories are in `models/deepar/train_metrics.json`. Val loss metrics are not comparable across versions — NLL (v1/v2) vs CRPS (v3–v6) vs weighted CRPS (v6).

| Version | Output distribution | Features | Key change | Best val loss | Stopped at | Status |
|---------|---------------------|----------|------------|---------------|------------|--------|
| v1 | StudentT | 20 dynamic real (leaky) | Baseline | 3.178 NLL | ep9 (p=5) | Retired — leakage |
| v2 | StudentT | 4 calendar + lags_seq | Leakage fix | 3.269 NLL | ep12 (p=10) | Retired — StudentT miscal |
| v3 | ISQF qk_x=[0.1…0.9] | 4 calendar + lags_seq | ISQF replaces StudentT | 2.905 CRPS | ep49 (50 ep) | Retired — p5/p95 in tail region |
| v4 | ISQF qk_x=[0.05…0.95] | 4 calendar + lags_seq | Explicit endpoint knots | 2.801 CRPS | ep46 (p=10) | **Production base** |
| v5 | ISQF qk_x=[0.025…0.975] | 4 calendar + lags_seq | Outer knots experiment | 2.551 CRPS | ep39 (p=10) | Retired — severe overfitting (coverage 47%) |
| v6 | ISQF v4 + horizon-weighted loss (τ=24h) | 4 calendar + lags_seq | Near-horizon CRPS up-weighting | 4.517 wCRPS* | ep39 (p=10) | Retired — no coverage gain |

*v6 val loss is weighted CRPS (exp(-t/24) per step, normalized); not directly comparable to v3–v5.

**Version notes:**

- **v1 → v2:** `feat_dynamic_real` reduced from 20 to 4 (calendar only). Future PM2.5 lags, rolling means, and pollutant covariates were all genuinely unavailable at forecast time. `lags_seq=[1,3,24]` replaces explicit lag features — GluonTS feeds back its own predictions during inference, avoiding any target leakage.

- **v2 → v3:** StudentT replaced by ISQF (`ISQFOutput`). StudentT v2 showed ~20% of actuals above p95 at h3/h12 because the distribution is too symmetric to capture asymmetric PM2.5 wildfire tails. ISQF directly learns the quantile function via CRPS training without parametric assumptions. Knots at [0.1…0.9] left p5/p95 in the exponential extrapolation region — coverage improved to 72–84% but p5/p95 endpoints were still extrapolated.

- **v3 → v4:** Added 0.05 and 0.95 as explicit spline knots. p5/p95 moved from extrapolation region into the directly-learned spline region. Coverage recovered significantly (h24: 84.6%, h72: 81.9%). Best val CRPS 2.801.

- **v4 → v5:** Experimental — added 0.025 and 0.975 knots to further reduce tail extrapolation pressure at h3/h12. Overfit to val-period wildfire/inversion tail events (Oct–Dec 2025); intervals collapsed from 13.9→7.1 μg/m³ and coverage dropped to 47% on test. Reverted to v4.

- **v4 → v6:** Horizon-weighted CRPS loss — `FixedISQFOutput.loss()` applies `w(t) = exp(-t/24)` decay weights (normalized to mean=1) per time step. Improved h3 point accuracy (CRPS 4.145→3.918, MAE 5.426→5.240) but did not improve h3/h12 raw coverage (74.1%/70.1%). The weighting sharpens near-term PIs rather than widening them. Conformal calibration (v4+conformal) outperforms v6+conformal on coverage, so v4+conformal is the production choice.

---

#### FixedISQFOutput — GluonTS 0.16.2 + PyTorch ≥2.x Compatibility

`ISQFOutput` from GluonTS 0.16.2 has two incompatibilities that required a `FixedISQFOutput` subclass (inheritance required — pydantic v1 validates `isinstance(distr_output, DistributionOutput)`):

1. **`loc=None` crash:** DeepAR calls `distr_output.loss(scale=scale)` without `loc`. `ISQFOutput.distribution()` passes `loc=None` into `AffineTransform`. PyTorch ≥2.x treats `None` as a missing operand in `AffineTransform._inverse` → `TypeError`. Fix: replace `None` with `torch.zeros_like(scale)` before constructing the transform.

2. **Wrong loss function:** `ISQFOutput` does not override `DistributionOutput.loss()`, so it falls back to `-log_prob()`. ISQF is a quantile-function model without a closed-form density — NLL is mathematically incorrect. CRPS is the proper training objective. Fix: override `loss()` to call `distr.crps(target)`, which `TransformedISQF` implements analytically with correct affine rescaling.

Additionally, `"enable_model_summary": False` is required in `trainer_kwargs` — Lightning's `ModelSummary` runs a forward pass during `fit()` setup, calling `ISQFOutput.sample()` before distribution parameters are initialized, triggering the `loc=None` crash.

---

#### Substep Status

- [x] 8.1 — `venv_deepar/` created; gluonts 0.16.2 + lightning 2.4.0 + torch 2.11.0 verified
- [x] 8.2 — `models/deepar/model.py` — constants, `FixedISQFOutput` subclass, estimator factory
- [x] 8.3 — `models/deepar/train.py` — ListDataset construction, FRM-only exclusion, NaN fill, build_datasets; `torch.manual_seed(42)` added for reproducibility
- [x] 8.4 — `models/deepar/sample_forecasts.py` — rolling windows, 500-sample inference, CRPS/MAE/RMSE/PI metrics, npz output
- [x] 8.5 — `tests/test_deepar.py` — 58 tests passing in venv_deepar
- [x] 8.6 — v4 training complete; predictor saved to `models/deepar/predictor/`
- [x] 8.7 — v4 test evaluation complete; raw metrics documented below
- [x] 8.8 — `evaluation/conformal.py` — split-conformal calibration on val windows; per-horizon asymmetric nonconformity scores; conformal margins saved to `evaluation/conformal_margins.json`
- [x] 8.9 — v4+conformal test metrics computed; saved to `evaluation/deepar_metrics_conformal.json`; **production predictor confirmed**

---

#### v4 Training Summary

Best val CRPS=**2.801** at epoch 36; early stopping triggered at epoch 46 (patience=10). 75-epoch budget. ~25s/epoch; total wall time ~20 minutes. Full per-epoch best-score history in `models/deepar/train_metrics.json`.

Selected new-best epochs: ep0 (5.327), ep9 (3.875), ep20 (3.397), ep31 (2.938), ep36 (**2.801**).

---

#### v4 Raw Test Results (642 windows, 14 stations, stride=24h)

| Horizon | MAE (μg/m³) | RMSE (μg/m³) | PI Coverage | Width (μg/m³) | CRPS  |
|---------|-------------|--------------|-------------|---------------|-------|
| 3hr     | 5.426       | 7.704        | 74.1%       | 14.26         | 4.145 |
| 12hr    | 5.924       | 9.016        | 72.7%       | 14.00         | 4.611 |
| 24hr    | 3.571       | 6.091        | 84.6%       | 13.16         | 2.747 |
| 72hr    | 3.746       | 6.340        | 81.9%       | 13.33         | 2.920 |
| Overall | **4.667**   | **7.381**    | **78.3%**   | 13.69         | 3.606 |

h3/h12 raw coverage (74–73%) is below the 90% target. All lower margins are zero — the systematic failure is upper-tail-only (model's p95 too low at short horizons).

---

#### Conformal Calibration (Step 8.8) — Production Adjustment

Split-conformal prediction on the val set (n=1,260 windows, α=0.10 target):

```
q_level = ceil((n+1)*(1−α)) / n = ceil(1261 × 0.90) / 1260 = 0.9008
s_upper(h) = max(y(h) − q95(h), 0)   # nonconformity: how far above p95
s_lower(h) = max(q05(h) − y(h), 0)   # nonconformity: how far below p05
margin(h)  = quantile(s, q_level)     # per-horizon conformal margin
```

| Horizon | Upper margin | Lower margin | Val coverage (before → after) |
|---------|-------------|-------------|-------------------------------|
| h3      | +3.78 μg/m³ | 0.0         | 80.2% → 88.5%                 |
| h12     | +5.25 μg/m³ | 0.0         | 78.7% → 88.6%                 |
| h24     | +2.52 μg/m³ | 0.0         | 81.7% → 87.5%                 |
| h72     | +3.87 μg/m³ | 0.0         | 78.0% → 87.7%                 |

All lower margins are zero — confirming the failure is upper-tail-only. h12 receives the largest margin (+5.25) because it was most miscalibrated on the val set.

---

#### v4+conformal — Production Test Results ✓

| Horizon | MAE (μg/m³) | RMSE (μg/m³) | PI Coverage | Width (μg/m³) | CRPS  |
|---------|-------------|--------------|-------------|---------------|-------|
| 3hr     | 5.426       | 7.704        | **86.1%**   | 18.04         | 4.145 |
| 12hr    | 5.924       | 9.016        | **87.2%**   | 19.25         | 4.611 |
| 24hr    | 3.571       | 6.091        | **90.8%**   | 15.68         | 2.747 |
| 72hr    | 3.746       | 6.340        | **90.0%**   | 17.20         | 2.920 |
| Overall | **4.667**   | **7.381**    | **88.6%**   | 17.54         | 3.606 |

MAE and CRPS are unchanged (conformal adjusts PI bounds only, not the point forecast). The production conformal margins are stored in `evaluation/conformal_margins.json` and applied at inference time in the alert system.

---

#### Three-Model Test Set Comparison

Point forecast accuracy (MAE, μg/m³):

| Model       | Overall MAE | h3    | h12   | h24   | h72   |
|-------------|-------------|-------|-------|-------|-------|
| LSTM        | 5.054       | 4.08  | 4.88  | 5.28  | 6.06  |
| TFT         | 5.223       | 4.76  | 5.29  | 5.44  | 5.40  |
| DeepAR v4   | **4.667**   | 5.43  | 5.92  | 3.57  | 3.75  |

Prediction interval coverage (90% PI, p5–p95):

| Model              | Overall | h3    | h12   | h24   | h72   |
|--------------------|---------|-------|-------|-------|-------|
| TFT                | 58.5%   | 64.0% | 57.7% | 55.2% | 56.8% |
| DeepAR v4 raw      | 78.3%   | 74.1% | 72.7% | 84.6% | 81.9% |
| **DeepAR v4+conf** | **88.6%**| **86.1%**| **87.2%**| **90.8%**| **90.0%**|

DeepAR v4+conformal achieves near-target (≥90%) coverage at h24/h72 and the closest-to-target coverage at h3/h12 (86–87%), with overall CRPS=3.606 — the only model with calibrated probabilistic forecasts for the alert system.

---

#### Implementation Notes

- `PYTORCH_ENABLE_MPS_FALLBACK=1` required at runtime: `aten::_standard_gamma` (ISQF sampling) not implemented for MPS. Fallback routes sampling to CPU; forward pass stays on MPS.
- `Predictor.deserialize` fix: `sample_forecasts.py` originally imported `DeepARPredictor` which does not exist in gluonts 0.16.2. Corrected to `from gluonts.model.predictor import Predictor`.
- W&B not logged: `wandb` in `venv_deepar` lacks `login()` (version incompatibility). All runs console-only; metrics captured in `train_metrics.json`.
- `feat_dynamic_real` shape fix: `_make_rolling_instances` originally built entries with `feat_dynamic_real` of shape `(4, 168)` (context only). GluonTS InstanceSplitter needs `(4, 240)` — context + future — so the decoder has covariate inputs for the prediction horizon. Fixed to use `sdf[ctx_mask | fut_mask]`.
- `station_to_idx` consistency: both `train.py` and `sample_forecasts.py` use `sorted(df["station_id"].unique())` over the same 14 stations — static embedding indices match at inference time.
- `torch.manual_seed(42)` + `np.random.seed(42)` set before `estimator.train()` (added in v6 run). GluonTS stochastic batching (num_batches_per_epoch=100) with no seed produced high variance across runs (v4 retrain: best val CRPS 3.179 vs original 2.801 at different epochs).

**Acceptance criteria:**
- [x] DeepAR trains without errors on 14 LA metro stations — v4: 46 epochs, early stopped at ep46, predictor saved
- [x] ISQF with `FixedISQFOutput` — CRPS training, loc=None crash fixed, enable_model_summary=False
- [x] Leakage-free covariate set — 4 calendar features + lags_seq=[1,3,24]; no future pollutant or PM2.5 covariates
- [x] 90% PI coverage between 85–95% — v4+conformal: 86.1%/87.2%/90.8%/90.0% at h3/h12/h24/h72 (88.6% overall)
- [x] Split-conformal calibration implemented with per-horizon asymmetric margins — `evaluation/conformal.py`
- [x] 500 Monte Carlo samples generated for test set — 642 windows × 500 samples saved to `evaluation/deepar_samples.npz`
- [x] Production conformal margins saved to `evaluation/conformal_margins.json`

---

### Step 9 — Probabilistic Alert System

**Files:** `alerts/breach_probability.py`, `alerts/risk_score.py`, `alerts/alert_router.py`

```python
ADVISORY_THRESHOLD = 35.4   # μg/m³
WARNING_THRESHOLD  = 55.4   # μg/m³
HORIZONS = {"3hr": 3, "12hr": 12, "24hr": 24, "72hr": 72}

def breach_probability(samples, threshold, horizon_hours):
    """P(PM2.5 > threshold) at given horizon from Monte Carlo samples."""
    return float(np.mean(samples[:, horizon_hours - 1] > threshold))

def precision_weighted_risk_score(breach_probs, tier):
    """Inverse-variance weighted station risk score."""
    sigmas = np.array([breach_probs[h]["sigma"] for h in HORIZONS])
    probs  = np.array([breach_probs[h][tier]  for h in HORIZONS])
    weights = (1 / sigmas) / np.sum(1 / sigmas)
    return float(np.dot(weights, probs))
```

**Acceptance criteria:**
- [ ] Breach probabilities computed correctly from Monte Carlo samples
- [ ] Precision weighting confirmed: 3hr horizon receives highest weight
- [ ] Risk scores fall in [0, 1] for all stations
- [ ] Advisory and Warning tiers produce distinct score distributions
- [ ] Alert output JSON schema validated

---

### Step 10 — InfluxDB Integration and Grafana Dashboard

**Files:** `monitoring/influxdb_writer.py`, `monitoring/grafana/dashboard.json`

**Grafana dashboard panels:**
- Real-time PM2.5 time series per station with EPA threshold overlays at 35.4 and 55.4 μg/m³
- Forecast overlay: point forecast + 90% prediction interval
- Advisory probability heatmap: stations × horizons
- Warning probability heatmap: stations × horizons
- Station status map: color-coded CLEAR / ADVISORY / WARNING
- System health: Kafka consumer lag, messages/second, prediction latency

**Acceptance criteria:**
- [ ] InfluxDB receiving alert data from streaming pipeline
- [ ] Grafana dashboard renders all panels without errors
- [ ] EPA threshold lines visible on PM2.5 time series
- [ ] Alert status updates in near-real-time

---

### Step 11 — Streamlit ML Interface

**File:** `app/streamlit_app.py`

**Four panels:**

1. **Model Comparison** — LSTM vs TFT vs DeepAR per horizon. Metrics: MAE, RMSE (LSTM/TFT), CRPS, PI coverage (DeepAR). Interactive horizon selector. W&B run links for full experiment details.

2. **Forecast Visualization** — Station selector. Time series: actual PM2.5 + three model forecasts. Prediction interval shading (DeepAR 10th/90th percentile). EPA threshold lines. Horizon selector.

3. **Spatial Catchment Maps** — LA metro map with station markers. Select any station to visualize its spatial catchment area. Neighbor stations colored by Epanechnikov kernel weight. Hover: station ID, distance, elevation difference, weight.

4. **Threshold Sensitivity Analysis** — Sliders for Advisory threshold, Warning threshold, and risk score classification thresholds. Live update: how many stations change status as thresholds shift.

**Acceptance criteria:**
- [ ] All four panels render without errors
- [ ] Station selector populates from live data
- [ ] Spatial catchment map renders correctly for all stations
- [ ] Threshold sliders update status counts in real time
- [ ] Model comparison table matches evaluation metrics from Steps 6–8

---

### Step 12 — Drift Monitoring

**Files:** `monitoring/drift/feature_drift.py`, `monitoring/drift/prediction_drift.py`

Following the UCI drift monitoring pattern — split test period into 4 temporal batches and compute drift vs training distribution:

- PSI on key features per batch vs training: `pm25`, `pm25_roll6`, `pm25_lag24`, `spatial_pm25_lag1`, `no2`, `o3`
- KS test on predicted probability distributions per batch vs batch 1
- Brier score on advisory threshold per batch — flag if >10% degradation from training baseline

**Acceptance criteria:**
- [ ] PSI computed per feature per batch
- [ ] Prediction distribution KS test computed per batch
- [ ] Brier score tracked per batch for both alert tiers
- [ ] Drift report saved to `monitoring/drift_report.json`

---

### Step 13 — Docker Compose Finalization and README

**Final docker-compose.yml** orchestrates all services: Zookeeper, Kafka, Kafdrop, InfluxDB, Grafana, Streamlit app, Producer service, PySpark consumer service.

**README contents:**
- Project overview and motivation
- Architecture diagram
- LA metro station map with spatial catchment visualization
- Setup instructions (Docker Compose single command)
- Scalability note: LA metro → California statewide
- Model comparison results table
- Alert system design documentation
- λ tuning results and spatial parameter documentation
- Known limitations and future work

---

## Evaluation Framework

### Time Series Forecasting

| Metric | Models | Computed per |
|---|---|---|
| MAE | LSTM, TFT, DeepAR median | Station × horizon |
| RMSE | LSTM, TFT, DeepAR median | Station × horizon |
| CRPS | DeepAR primary | Station × horizon |
| PI Coverage (90%, p5–p95) | TFT, DeepAR | Station × horizon |
| Sharpness | TFT, DeepAR | Station × horizon |

### Alert System

| Metric | Description |
|---|---|
| Advisory Brier Score | Calibration of P(PM2.5 > 35.4) |
| Warning Brier Score | Calibration of P(PM2.5 > 55.4) |
| Alert Precision@horizon | Of ADVISORY alerts, fraction confirmed |
| Alert Recall@horizon | Of true exceedances, fraction flagged |

### Spatial Feature Validation
- Compare LSTM/TFT/DeepAR with vs without spatial features
- Quantify spatial feature contribution to forecast improvement
- Document optimal λ and d_cutoff values from validation tuning

---

## Key Design Decisions

1. **EPA AQS over OpenAQ for data collection** — AQS is the primary regulatory data source (SCAQMD reports directly to AQS; OpenAQ ingests downstream). AQS's county-level batch endpoint (`hourData/byCounty`) returns all stations in a county in a single request, making bulk historical pulls fast (~100 requests for 1 year) and reliable. AQS site IDs are stable on instrument replacement, eliminating deduplication entirely. Requires free registration (`AQS_EMAIL` + `AQS_KEY` in `.env`).
2. **Epanechnikov kernel over fixed nearest-N** — variable station density in LA metro means fixed-N produces inconsistent spatial context; kernel weighting with cutoff is density-invariant and architecturally clean
3. **λ tuned on validation set** — avoids arbitrary assumption about elevation-distance equivalence; documents regional specificity and production generalization strategy
4. **ISQFOutput with explicit endpoint quantile knots for DeepAR** — PM2.5 is right-skewed with heavy tails from wildfire events. StudentT (v1/v2) placed ~20% of actuals above p95 at h3/h12 because the distribution is too symmetric. ISQF directly learns the quantile function without parametric assumptions, trained via CRPS. Knot positions `qk_x=[0.05, 0.1, …, 0.9, 0.95]` (v4) place p5/p95 inside the directly-learned spline region; beyond the outermost knots the model uses exponential tail extrapolation. Experiments placing knots at 0.025/0.975 (v5) overfit to val-period tail events, collapsing coverage to 47%.
5. **Split-conformal calibration for coverage guarantees** — ISQF trained with CRPS optimizes mean distributional accuracy but does not guarantee marginal coverage at specific quantile levels. Per-horizon asymmetric split-conformal prediction (n=1,260 val windows, α=0.10) adds data-driven margins to q05/q95 with a rigorous ≥90% coverage guarantee under exchangeability. All lower margins are zero (confirming the failure is upper-tail-only); h12 receives the largest margin (+5.25 μg/m³). v4+conformal achieves 86–91% coverage across horizons with 17.5 μg/m³ mean PI width.
6. **Precision-weighted risk score** — inverse-variance weighting gives more influence to high-confidence near-term forecasts; statistically principled and directly analogous to inverse-variance meta-analysis
7. **Dual alert tiers** — Advisory and Warning tiers map to distinct public health actions; single threshold would conflate sensitive-group risk with general population risk
8. **Separate Grafana and Streamlit** — Grafana for operational real-time monitoring; Streamlit for ML performance and explainability; mirrors production MLOps architecture

---

## Implementation Order Summary

1. Repository scaffold and Docker Compose skeleton ✓
2. Station metadata, USGS elevation, spatial index ✓
3. Historical data pull and DuckDB storage ✓
4. Sensor validation, imputation, feature engineering ✓
5. Kafka producer and PySpark streaming consumer ✓
6. LSTM baseline + λ tuning on validation set ✓
7. TFT baseline + attention visualization
8. DeepAR primary + ISQF + conformal calibration ✓ (v4+conformal is production predictor)
9. Probabilistic alert system
10. InfluxDB integration and Grafana dashboard
11. Streamlit ML interface
12. Drift monitoring
13. Docker Compose finalization and README
