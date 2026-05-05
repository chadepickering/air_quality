# Real-Time Air Quality Forecasting and Health Alert System

An end-to-end real-time environmental data pipeline that ingests streaming air quality sensor readings from monitoring stations across the LA metro area, forecasts PM2.5 concentrations at multiple time horizons using three time series models, and generates probabilistic public health alerts when predicted air quality is projected to breach EPA threshold levels.

**Type:** Independent production ML/AI project  
**Status:** Implementation in progress  
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
│  - Real-time feeds   │   │  - Spatial catchment maps  │
│  - Forecast overlay  │   │  - Threshold sensitivity   │
│  - Alert status      │   │  - Attention weights       │
│  - System health     │   │                            │
└──────────────────────┘   └────────────────────────────┘
```

---

## Quick Start

```bash
git clone https://github.com/chadepickering/air_quality_forecasting.git
cd air_quality_forecasting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pull station metadata and historical data
python -m ingestion.openaq_client
python -m ingestion.usgs_elevation

# Start infrastructure
docker compose up -d zookeeper kafka influxdb grafana

# Stream historical data and run forecasting pipeline
python -m streaming.producer
```

---

## Project Structure

```
air_quality/
├── alerts/
│   ├── alert_router.py             # CLEAR / ADVISORY / WARNING classification
│   ├── breach_probability.py       # P(PM2.5 > threshold) from Monte Carlo samples
│   ├── risk_score.py               # Precision-weighted station risk score
│   └── threshold_config.py         # EPA threshold definitions
├── app/
│   └── streamlit_app.py            # ML interface — model comparison, forecast viz, maps
├── data/
│   ├── metadata/                   # committed — station list + USGS elevations
│   ├── processed/                  # gitignored — parquet feature files
│   └── raw/                        # gitignored — OpenAQ API responses
├── evaluation/
│   ├── model_comparison.py         # LSTM vs TFT vs DeepAR comparison
│   ├── spatial_catchment_viz.py    # Epanechnikov kernel weight maps per station
│   └── threshold_sensitivity.py    # Risk score under different thresholds
├── ingestion/
│   ├── database.py                 # DuckDB schema and write helpers
│   ├── openaq_client.py            # OpenAQ REST API v3 wrapper + pagination
│   ├── station_registry.py         # Station metadata, spatial index, Epanechnikov weights
│   └── usgs_elevation.py           # One-time USGS elevation pull per station
├── models/
│   ├── deepar/
│   │   ├── evaluate.py             # CRPS, prediction interval coverage
│   │   ├── model.py                # DeepAR via GluonTS (StudentT output)
│   │   ├── sample_forecasts.py     # Monte Carlo sample generation (500 trajectories)
│   │   └── train.py
│   ├── lstm/
│   │   ├── evaluate.py             # MAE, RMSE per horizon
│   │   ├── model.py                # Two-layer LSTM, point forecast per horizon
│   │   └── train.py
│   └── tft/
│       ├── attention_viz.py        # Variable selection weights + attention patterns
│       ├── evaluate.py             # Quantile coverage evaluation
│       ├── model.py                # TFT via PyTorch Forecasting
│       └── train.py
├── monitoring/
│   ├── drift/
│   │   ├── feature_drift.py        # PSI on key features vs training distribution
│   │   └── prediction_drift.py     # Brier score and alert distribution monitoring
│   ├── grafana/
│   │   ├── alerts.json             # Grafana alert rules
│   │   └── dashboard.json          # Grafana dashboard config
│   └── influxdb_writer.py          # Stream predictions and alerts to InfluxDB
├── notebooks/
│   └── exploration.ipynb           # EDA, station selection, seasonality
├── streaming/
│   ├── consumer.py                 # PySpark Structured Streaming consumer
│   ├── feature_engineering.py      # Temporal and spatial feature computation
│   ├── producer.py                 # Multi-station Kafka producer
│   ├── sensor_validation.py        # Quality flagging and imputation
│   └── spatial_weights.py          # Epanechnikov kernel, composite distance metric
├── tests/
│   ├── test_alert_system.py
│   ├── test_risk_score.py
│   ├── test_sensor_validation.py
│   └── test_spatial_weights.py
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml              # Full multi-service orchestration
├── README.md
├── README_proj-plan.md             # Full implementation plan with step-by-step build log
└── requirements.txt
```

---

## Stack

| Component | Technology |
|---|---|
| Air quality data | OpenAQ REST API v3 |
| Elevation data | USGS National Elevation Dataset |
| Local storage | DuckDB |
| Message broker | Apache Kafka (Docker) |
| Stream processing | PySpark Structured Streaming |
| Time series DB | InfluxDB (Docker) |
| LSTM baseline | TensorFlow/Keras |
| TFT baseline | PyTorch Forecasting |
| DeepAR primary | GluonTS (PyTorch backend) |
| Experiment tracking | Weights & Biases |
| Operational dashboard | Grafana (Docker) |
| ML interface | Streamlit |
| Containerization | Docker Compose |

---

## Key Design Decisions

**Epanechnikov kernel over fixed nearest-N.** Variable station density in LA metro means fixed-N produces inconsistent spatial context across urban and rural stations. Kernel weighting with a distance cutoff is density-invariant and scales cleanly from LA metro to California statewide without architectural changes.

**λ tuned on validation set.** The composite distance metric combines haversine distance and elevation difference via a scaling parameter λ. Rather than assuming a fixed elevation-distance equivalence, λ is optimized on held-out stations, documenting regional specificity and production generalization strategy explicitly.

**StudentT output for DeepAR.** PM2.5 distributions are right-skewed with heavy tails from wildfire smoke events and temperature inversions. StudentT produces better-calibrated extreme-event prediction intervals than Gaussian output.

**Precision-weighted risk score.** Inverse-variance weighting gives more influence to high-confidence near-term forecasts — the 3-hour horizon is most confident and therefore most heavily weighted. This is statistically principled and directly analogous to inverse-variance meta-analysis.

**Dual alert tiers.** Advisory (>35.4 μg/m³) and Warning (>55.4 μg/m³) tiers map to distinct public health actions. A single threshold would conflate sensitive-group risk with general population risk.

**Separate Grafana and Streamlit.** Grafana for operational real-time monitoring; Streamlit for ML performance and explainability. Mirrors production MLOps architecture where operational dashboards and ML interfaces serve different audiences.
