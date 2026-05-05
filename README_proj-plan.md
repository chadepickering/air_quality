# Real-Time Air Quality Forecasting and Health Alert System

## Overview

An end-to-end real-time environmental data pipeline that ingests streaming air quality sensor readings from multiple monitoring stations across the LA metro area, forecasts PM2.5 concentrations at multiple time horizons using three time series models, and generates probabilistic public health alerts when predicted air quality is projected to breach EPA threshold levels. The system demonstrates production-grade streaming infrastructure, state-of-the-art probabilistic time series deep learning, spatial feature engineering, and operational monitoring — all on freely available public data with zero cloud cost.

**Total cost to run:** $0 (fully open-source stack)

---

## Architecture

```
OpenAQ REST API (LA metro monitoring stations)
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
| Primary source | OpenAQ REST API v3 |
| Endpoint | `https://api.openaq.org/v3/` |
| Development scope | LA metro area (South Coast AQMD network, 30+ stations) |
| Production extension | State of California (CARB network, 250+ stations) |
| Access | No credentials required |
| Format | JSON, paginated |
| Temporal resolution | Hourly |
| Primary target | PM2.5 (μg/m³) |
| Covariates | NO2, O3, PM10, CO |
| Elevation data | USGS National Elevation Dataset — one-time point query per station |

**Why LA metro:** South Coast AQMD operates one of the densest air quality monitoring networks in the world. The LA basin's geographic and meteorological complexity — ocean breeze, temperature inversions, wildfire smoke events, traffic corridors — creates rich temporal patterns that reward sophisticated modeling over simpler baselines.

**Scalability note:**

| Tier | Pattern | When to use |
|---|---|---|
| Development | OpenAQ API → local Kafka → local models | LA metro, portfolio demonstration |
| Staging | OpenAQ API → GCS → Kafka → Spark cluster | California statewide, 250+ stations |
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
│   │   └── station_elevations.csv
│   ├── processed/                  # gitignored
│   └── raw/                        # gitignored
├── evaluation/
│   ├── model_comparison.py
│   ├── spatial_catchment_viz.py
│   └── threshold_sensitivity.py
├── ingestion/
│   ├── database.py
│   ├── openaq_client.py
│   ├── station_registry.py
│   └── usgs_elevation.py
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
│   ├── feature_engineering.py
│   ├── producer.py
│   ├── sensor_validation.py
│   └── spatial_weights.py
├── tests/
│   ├── test_alert_system.py
│   ├── test_risk_score.py
│   ├── test_sensor_validation.py
│   └── test_spatial_weights.py
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── README.md
├── README_proj-plan.md
└── requirements.txt
```

---

## Stack

| Component | Tool | Cost |
|---|---|---|
| Air quality data | OpenAQ REST API v3 | Free |
| Elevation data | USGS National Elevation Dataset | Free |
| Local storage | DuckDB | Free |
| Message broker | Apache Kafka (Docker) | Free |
| Stream processing | PySpark Structured Streaming | Free |
| Time series DB | InfluxDB (Docker) | Free |
| LSTM baseline | TensorFlow/Keras | Free |
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
- Prediction Interval Coverage — what fraction of true values fall within the 90% PI
- Sharpness — mean width of prediction intervals (narrower is better, conditional on coverage)

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
pip install pyspark kafka-python
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

**Files:** `ingestion/openaq_client.py`, `ingestion/usgs_elevation.py`, `ingestion/station_registry.py`

**OpenAQ station pull for LA metro:**
```python
# ingestion/openaq_client.py
import requests

BASE_URL = "https://api.openaq.org/v3"

def fetch_la_stations() -> list[dict]:
    # South Coast AQMD bounding box
    # lat: 33.5 to 34.8, lon: -118.9 to -117.0
    params = {
        "bbox": "-118.9,33.5,-117.0,34.8",
        "parameters": "pm25",
        "limit": 100
    }
    response = requests.get(f"{BASE_URL}/locations", params=params)
    response.raise_for_status()
    return response.json()["results"]
```

**USGS elevation pull — one time only:**
```python
# ingestion/usgs_elevation.py
import requests

def get_elevation(lat: float, lon: float) -> float:
    url = "https://epqs.nationalmap.gov/v1/json"
    params = {"x": lon, "y": lat, "units": "Meters"}
    response = requests.get(url, params=params)
    return response.json()["value"]
```

**Spatial index and distance computation:**
```python
# ingestion/station_registry.py
from haversine import haversine
import numpy as np

D_CUTOFF_KM = 40.0

def composite_distance(s1: dict, s2: dict, lambda_param: float) -> float:
    d_haversine = haversine((s1["lat"], s1["lon"]), (s2["lat"], s2["lon"]))
    delta_elev = abs(s1["elevation_m"] - s2["elevation_m"])
    return np.sqrt(d_haversine**2 + lambda_param * delta_elev**2)

def epanechnikov_weight(d: float, d_cutoff: float = D_CUTOFF_KM) -> float:
    if d >= d_cutoff:
        return 0.0
    return max(0.0, 1 - (d / d_cutoff)**2)

def build_spatial_neighbor_index(
    stations: list[dict],
    lambda_param: float,
    d_cutoff: float = D_CUTOFF_KM
) -> dict[str, list[tuple[str, float]]]:
    index = {}
    for s in stations:
        neighbors = []
        for other in stations:
            if other["id"] == s["id"]:
                continue
            d = composite_distance(s, other, lambda_param)
            w = epanechnikov_weight(d, d_cutoff)
            if w > 0:
                neighbors.append((other["id"], w))
        total = sum(w for _, w in neighbors)
        index[s["id"]] = [(sid, w/total) for sid, w in neighbors]
    return index
```

**λ tuning:** Use λ=0.1 as initial value during development. Tune on held-out stations after LSTM baseline is established (Step 6).

**Acceptance criteria:**
- [ ] LA metro stations pulled and stored to `data/metadata/stations.csv`
- [ ] Elevation enriched and stored to `data/metadata/station_elevations.csv`
- [ ] Spatial neighbor index computed for all stations at λ=0.1
- [ ] Visual inspection of neighbor assignments makes geographic sense
- [ ] At least 20 stations with complete PM2.5 coverage identified

---

### Step 3 — Historical Data Pull and DuckDB Storage

**Files:** `ingestion/database.py`, `ingestion/openaq_client.py`

**DuckDB schema:**
```python
def initialize_database(db_path: str = "data/processed/aq.duckdb"):
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_readings (
            station_id VARCHAR,
            parameter VARCHAR,
            value FLOAT,
            unit VARCHAR,
            timestamp TIMESTAMP,
            quality_flag INTEGER,    -- 0=valid, 1=suspect, 2=invalid
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (station_id, parameter, timestamp)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS processed_features (
            station_id VARCHAR,
            timestamp TIMESTAMP,
            pm25 FLOAT,
            no2 FLOAT, o3 FLOAT, pm10 FLOAT, co FLOAT,
            hour_of_day INTEGER, day_of_week INTEGER,
            month INTEGER, is_weekend BOOLEAN,
            pm25_roll3 FLOAT, pm25_roll6 FLOAT, pm25_roll24 FLOAT,
            pm25_lag1 FLOAT, pm25_lag3 FLOAT, pm25_lag24 FLOAT,
            spatial_pm25_lag1 FLOAT, spatial_pm25_lag3 FLOAT,
            spatial_pm25_roll6 FLOAT, spatial_no2_lag1 FLOAT,
            spatial_o3_lag1 FLOAT, spatial_elev_diff FLOAT,
            PRIMARY KEY (station_id, timestamp)
        )
    """)
    return con
```

**Acceptance criteria:**
- [ ] At least 12 months of hourly PM2.5 data pulled for all LA metro stations
- [ ] NO2, O3, PM10 covariates pulled for same stations and period
- [ ] Raw data stored in DuckDB with quality flags
- [ ] Data completeness report: % valid readings per station per parameter
- [ ] Stations with <70% PM2.5 completeness flagged for exclusion

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
- Missing 4–24 hours: same-hour-of-day median from prior 7 days
- Missing >24 hours: station excluded from spatial features for that period, flag propagated downstream

**Acceptance criteria:**
- [ ] Sensor validation correctly flags known outliers in historical data
- [ ] Imputation fills gaps without introducing artifacts
- [ ] All temporal features computed correctly — spot check 10 stations
- [ ] Spatial features computed correctly — verify neighbor weights sum to 1
- [ ] Processed features written to DuckDB `processed_features` table

---

### Step 5 — Kafka Producer and PySpark Streaming Consumer

**Files:** `streaming/producer.py`, `streaming/consumer.py`

**Multi-station Kafka producer:** Replays historical data at configurable speed (default: 1 simulated hour = 1 real minute). One Kafka message per station per timestamp, keyed by `station_id` for partition locality.

**PySpark Structured Streaming consumer:** Reads from `raw_air_quality`, applies validation and feature engineering, writes processed features to `processed_air_quality` topic.

**Acceptance criteria:**
- [ ] Producer replays 30 days of historical data without errors
- [ ] All 20+ stations stream simultaneously as separate Kafka partitions
- [ ] PySpark consumer processes messages in near-real-time
- [ ] Processed features written to `processed_air_quality` topic
- [ ] Kafdrop shows correct topic partitioning and message throughput

---

### Step 6 — LSTM Baseline

**Files:** `models/lstm/model.py`, `models/lstm/train.py`, `models/lstm/evaluate.py`

**Architecture:** Two-layer LSTM (64 units → 32 units), 24hr lookback, point forecast output for 4 horizons (3hr, 12hr, 24hr, 72hr).

**Train/validation/test split:**
- Train: first 18 months
- Validation: months 19–21 (used for λ tuning and hyperparameter selection)
- Test: final 3 months (held out until final evaluation)

**λ tuning on validation set:** Grid search over λ ∈ {0.05, 0.10, 0.15, 0.20} and d_cutoff ∈ {30, 40, 50} km. Select combination minimizing validation MAE. Recompute spatial features with tuned parameters before TFT and DeepAR.

**Acceptance criteria:**
- [ ] LSTM trains without errors on processed feature set
- [ ] λ tuned on validation set — optimal value documented
- [ ] Spatial features recomputed with tuned λ
- [ ] Validation MAE < 8 μg/m³ at 3hr horizon
- [ ] W&B run logged with training curves and per-horizon metrics

---

### Step 7 — TFT Baseline

**Files:** `models/tft/model.py`, `models/tft/train.py`, `models/tft/attention_viz.py`

TFT via PyTorch Forecasting. Key capabilities: variable selection networks (learns which features matter per station), multi-head attention (identifies which historical timesteps matter at each horizon), quantile regression (10th/50th/90th percentile forecasts for PI coverage evaluation).

**Key outputs to visualize:**
- Variable selection weights: which features TFT finds most informative per station
- Attention patterns: which historical hours most influence each horizon
- Quantile forecasts: 90% PI coverage evaluation

**Acceptance criteria:**
- [ ] TFT trains without errors
- [ ] TFT outperforms LSTM on validation MAE at 12hr and 24hr horizons
- [ ] Variable selection weights visualized and saved
- [ ] Attention patterns visualized for representative stations
- [ ] 90% PI coverage between 85–95%

---

### Step 8 — DeepAR Primary Model

**Files:** `models/deepar/model.py`, `models/deepar/train.py`, `models/deepar/sample_forecasts.py`

DeepAR via GluonTS (PyTorch backend). Autoregressive recurrent model outputting full predictive distributions via Monte Carlo trajectories. StudentT output distribution chosen for heavy-tailed PM2.5 behavior during wildfire and inversion events.

```python
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import StudentTOutput

estimator = DeepAREstimator(
    freq="H",
    prediction_length=72,
    context_length=168,       # 7-day lookback
    distr_output=StudentTOutput(),
    num_feat_dynamic_real=15,
    num_feat_static_cat=1,    # station ID
    trainer_kwargs={"max_epochs": 50}
)
```

**Monte Carlo sample generation:** 500 trajectories per station per forecast. Samples feed directly into the alert system's breach probability computation.

**Acceptance criteria:**
- [ ] DeepAR trains without errors on all LA metro stations
- [ ] CRPS lower than LSTM and TFT equivalent
- [ ] 90% prediction interval coverage between 85–95%
- [ ] StudentT distribution produces wider intervals during high-PM2.5 periods
- [ ] 500 Monte Carlo samples generated for test set

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
| PI Coverage (90%) | TFT, DeepAR | Station × horizon |
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

1. **Epanechnikov kernel over fixed nearest-N** — variable station density in LA metro means fixed-N produces inconsistent spatial context; kernel weighting with cutoff is density-invariant and architecturally clean
2. **λ tuned on validation set** — avoids arbitrary assumption about elevation-distance equivalence; documents regional specificity and production generalization strategy
3. **StudentT output for DeepAR** — PM2.5 is right-skewed with heavy tails from wildfire events; StudentT produces better-calibrated extreme event prediction intervals than Gaussian
4. **Precision-weighted risk score** — inverse-variance weighting gives more influence to high-confidence near-term forecasts; statistically principled and directly analogous to inverse-variance meta-analysis
5. **Dual alert tiers** — Advisory and Warning tiers map to distinct public health actions; single threshold would conflate sensitive-group risk with general population risk
6. **Separate Grafana and Streamlit** — Grafana for operational real-time monitoring; Streamlit for ML performance and explainability; mirrors production MLOps architecture

---

## Implementation Order Summary

1. Repository scaffold and Docker Compose skeleton ✓
2. Station metadata, USGS elevation, spatial index
3. Historical data pull and DuckDB storage
4. Sensor validation, imputation, feature engineering
5. Kafka producer and PySpark streaming consumer
6. LSTM baseline + λ tuning on validation set
7. TFT baseline + attention visualization
8. DeepAR primary + Monte Carlo sample generation
9. Probabilistic alert system
10. InfluxDB integration and Grafana dashboard
11. Streamlit ML interface
12. Drift monitoring
13. Docker Compose finalization and README
