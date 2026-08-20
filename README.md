# Pearls AQI Predictor

3-day-ahead Air Quality Index forecast for Lahore. Built as a serverless Feature / Training / Inference (FTI) pipeline for the 10Pearls Data Science Internship.

Pollution comes from the OpenWeather Air Pollution API. Weather comes from Open-Meteo. Features and models live in Hopsworks. Inference is FastAPI; the dashboard is Streamlit.

## Architecture

```
OpenWeather (pollution) ─┐
                          ├─▶ Feature Pipeline ──▶ Hopsworks Feature Store
Open-Meteo (weather)    ─┘   (GitHub Actions, hourly)        │
                                                             ▼
                                                      Training Pipeline ──▶ Hopsworks Model Registry
                                                      (GitHub Actions, daily)         │
                                                                                      ▼
                                                                               FastAPI (inference)
                                                                                      │
                                                                                      ▼
                                                                               Streamlit Dashboard
```

## Requirements

- Python 3.11
- An [OpenWeather](https://openweathermap.org/api) API key (air pollution history)
- A [Hopsworks](https://www.hopsworks.ai/) project (feature store + model registry)
- Open-Meteo needs no key

## Setup

```bash
git clone <this-repo>
cd aqi-predictor
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set at least:

```
OPENWEATHER_API_KEY=...
HOPSWORKS_API_KEY=...
HOPSWORKS_PROJECT_NAME=...
FEATURE_GROUP_VERSION=4
```

Optional values (`CITY_NAME`, coordinates, `DATA_START_DATE`, `TRAIN_START_DATE`) are documented in `.env.example`. Defaults target Lahore.

## How to use

Run commands from the repository root.

### 1. Fetch raw history

```bash
python -m src.feature_pipeline raw-snapshot
```

Writes `data/raw/aqi_raw_merged.parquet` (pollution + weather, validated). No feature engineering and no Hopsworks write.

### 2. Explore the data

Open `notebooks/01_eda.ipynb` after the raw snapshot exists.

### 3. Push features to Hopsworks

One-time (or whenever you rebuild the feature group):

```bash
python -m src.feature_pipeline push-features
python -m src.utils.hopsworks_utils
```

`push-features` builds the engineered table from the local parquet and inserts it into `aqi_features` v4. The second command creates the feature view used by training and the API.

Hourly updates (lookback window, skip-if-exists, upsert missing hours):

```bash
python -m src.feature_pipeline
```

Historical backfill into Hopsworks (fetch + engineer + insert):

```bash
python -m src.feature_pipeline backfill
```

### 4. Train models

```bash
python -m src.training_pipeline
```

Trains the model set per horizon (24h / 48h / 72h), scores against a persistence baseline, and registers `aqi_forecaster_{24,48,72}h` in Hopsworks only when a model beats that baseline on RMSE, MAE, and R².

GPU training can also be run from `notebooks/02_training.ipynb`. Metrics and SHAP plots land in `reports/`.

### 5. Serve locally

Terminal 1:

```bash
uvicorn api.main:app --reload
```

Terminal 2:

```bash
streamlit run app/Home.py
```

`GET /predict` returns the 3-day forecast. The Streamlit app shows EDA, forecast, alerts, and model performance.

### 6. Docker

```bash
docker compose up --build
```

Set `API_BASE_URL` if the dashboard should call a non-default API host (see `.env.example`).

### 7. GitHub Actions

Add these **repository secrets**: `OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`.

| Workflow | Schedule | Command |
|---|---|---|
| `.github/workflows/feature_pipeline.yml` | Hourly | `python -m src.feature_pipeline` |
| `.github/workflows/training_pipeline.yml` | Daily 02:00 UTC | `python -m src.training_pipeline` |

Both can also be started from the Actions tab (`workflow_dispatch`). Feature-group version is pinned to **4** in the workflow files.

## Tests

```bash
pytest tests/
```

## Repository layout

```
aqi-predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # hourly GitHub Action
│       └── training_pipeline.yml   # daily GitHub Action
├── api/
│   ├── Dockerfile
│   ├── main.py                     # FastAPI /predict
│   └── requirements.txt
├── app/
│   ├── Dockerfile
│   ├── Home.py                     # Streamlit entrypoint
│   ├── requirements.txt
│   └── pages/
│       ├── 1_EDA.py
│       ├── 2_Forecast.py
│       ├── 3_Alerts.py
│       └── 4_Model_Performance.py
├── data/                           # local parquet (gitignored); created by raw-snapshot
├── notebooks/
│   ├── README.md
│   ├── 01_eda.ipynb
│   └── 02_training.ipynb
├── reports/
│   └── README.md                   # SHAP plots and comparison CSVs land here
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── feature_pipeline.py         # raw-snapshot | backfill | push-features | hourly
│   ├── training_pipeline.py
│   ├── train_prophet.py
│   ├── train_lstm.py
│   ├── train_gru.py
│   └── utils/
│       ├── __init__.py
│       ├── aqi_calculation.py      # US EPA AQI from PM2.5
│       ├── data_fetch.py           # OpenWeather + Open-Meteo + merge
│       ├── data_validation.py
│       ├── evaluation.py           # RMSE/MAE/R², persistence, shrinkage
│       ├── feature_engineering.py
│       ├── hopsworks_utils.py
│       ├── optuna_tuning.py
│       ├── raw_io.py               # local raw parquet read/write
│       └── sequences.py            # LSTM/GRU windows
├── tests/
│   ├── test_aqi_calculation.py
│   ├── test_data_validation.py
│   ├── test_evaluation.py
│   ├── test_feature_engineering.py
│   └── test_feature_pipeline.py
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── README.md
└── requirements.txt
```

## Live demo

_(Add the deployed URL here.)_

## Report

Write-up and metric tables: `reports/final_report.md`.
