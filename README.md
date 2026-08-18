# Pearls AQI Predictor

End-to-end, serverless AQI (Air Quality Index) forecasting system for Lahore — 3-day-ahead
forecast, built for the 10Pearls Data Science Internship using a Feature/Training/Inference
(FTI) pipeline architecture.

## Architecture

```
OpenWeather (pollution/AQI) ─┐
                              ├─▶ Feature Pipeline ──▶ Hopsworks Feature Store
Open-Meteo (weather)        ─┘   (GitHub Actions,           │
                                   hourly)                    ▼
                                                        Training Pipeline ──▶ Hopsworks Model Registry
                                                        (GitHub Actions,             │
                                                         daily)                      ▼
                                                                              FastAPI (inference)
                                                                                      │
                                                                                      ▼
                                                                              Streamlit Dashboard
```

## Final decisions (why things are built this way)

**Data sources — one source per variable, always:**
- **Pollution / AQI** (the prediction target) → **OpenWeather Air Pollution API** only — current, forecast, and historical. Never mixed with another pollution source, since different sensors/models measure pollution differently and mixing would inject noise into the label.
- **Weather** (temperature/humidity/wind) → **Open-Meteo** only — current AND historical, in *both* the hourly pipeline and backfill. Free, no API key, no card. Using OpenWeather's live weather but Open-Meteo's historical weather would create train/serving skew (the exact problem a feature store exists to prevent) — so it's Open-Meteo everywhere, consistently.
- Both sources use the **same lat/lon** (`config.LATITUDE` / `config.LONGITUDE`, both from Lahore's coordinates) and are **normalized to UTC** before merging — see `data_fetch.py`.
- Both start from the **same date** (`config.DATA_START_DATE`, defaults to `2020-11-27`, the start of OpenWeather's free historical archive) so the two dataframes align from row zero.

**How much data:** the full available history (~5.5 years, Nov 2020–present) is fetched and kept as the raw snapshot. **Training, however, starts at `config.TRAIN_START_DATE` (default `2025-04-04`)** — OpenWeather's archive changes character on that date: mean hourly |AQI change| collapses from ~46 to ~4.5 and never recovers, and the same break appears in PM2.5. Everything before it is effectively a different data-generating process, so models trained across the break learned a volatility the present no longer has and lost to the persistence baseline at every horizon. Restricting training to the post-break regime is what made them win. The earlier history is still fetched and stored — set `TRAIN_START_DATE=` (empty) to train on all of it and reproduce the failure.

**Hourly grid is made strict before any shifting** (`feature_engineering.to_hourly_grid()`): lags, rolling windows, change rates and targets all shift by ROW POSITION, which only equals hours if no hour is missing. The raw history has ~3,700 missing hours across 415 outages, so on the gappy frame `shift(-24)` actually spanned 24 hours for only 87% of rows (up to 264 hours for the rest) — those rows were trained against the wrong future. The frame is now reindexed onto a complete hourly grid first; outages of ≤6 hours are interpolated, longer ones stay missing and those rows are dropped rather than invented. This is the difference between feature group v3 and **v4**.

**The AQI target — computed, not taken from OpenWeather directly:** OpenWeather's own `main.aqi` field is NOT a continuous AQI — it's a coarse 1-5 category (1=Good ... 5=Very Poor), documented at openweathermap.org/api/air-pollution. Training regression models (RMSE/MAE/R²) against a 5-value categorical field wouldn't be meaningful, and doesn't match what "AQI" means in the brief (a continuous number, like the "82" in the brief's own example screenshot). So `aqi` throughout this project is computed from the raw PM2.5 concentration using the real **US EPA AQI formula** (`src/utils/aqi_calculation.py`), verified against the EPA's 2024-revised breakpoint table and its own published worked examples. OpenWeather's original 1-5 field is kept as a separate `openweather_aqi_category` column for reference only — never used as a feature or target. (This module also had a real bug caught during testing — a raw float like 9.0989 fell in a gap between adjacent EPA breakpoints and produced a nonsensical negative AQI; fixed by truncating to 1 decimal place first, per the official EPA method, and locked in with a regression test covering all 4001 possible truncated values from 0-400.)

**Data validation** (`src/utils/data_validation.py`): runs on every merged (pollution + weather) batch, in both the hourly pipeline and backfill, BEFORE feature engineering. Checks required columns exist, drops duplicate timestamps, drops rows with nulls or out-of-range values (sensor glitches/API errors) — one bad hour is dropped, not the whole batch.

**Train/val/test split — season-aligned, not a blind percentage cut:**
`training_pipeline.chronological_split()` reserves the most recent ~1 year for test and the year before that for validation, with both boundaries snapped to **1 June** (not 1 Jan) so Lahore's Oct–Jan smog season always sits safely inside a partition instead of being cut in half at a Dec 31/Jan 1 boundary. This guarantees every partition — train, val, and test — contains a full seasonal cycle. Falls back to a simple 70/15/15 split automatically if there isn't enough history yet, which is the active path while the post-regime-break window is under ~2.5 years. One consequence to read honestly: the fallback's test window is a low-variance summer stretch with no smog season, which makes R² a harsh metric there (a small variance denominator can leave even the persistence baseline at a negative R²), so RMSE and MAE carry more signal than R² in that report.

**Predicted deltas are shrunk toward zero by a factor fitted on validation** (`evaluation.fit_delta_shrinkage()`): λ=0 is exactly the persistence baseline and λ=1 is the untouched model, so the search interpolates between them and each delta model gets a second, shrunk candidate in the comparison at no extra training cost. It exists because predicted deltas are noisy and an over-confident delta costs the most on the many hours where AQI barely moves — the hours where persistence is near-exact. The winning λ is stored with the registered model and re-applied at serve time, so the served prediction is the one that passed the gate.

**Evaluation — three layers, not just one RMSE number:**
1. Standard RMSE/MAE/R² per model (Persistence, Prophet, Ridge, Random Forest, XGBoost, LightGBM, LSTM, GRU) → `reports/model_comparison.csv`
2. **Stratified**: smog season vs. normal season, on the winning model → `reports/model_comparison_stratified.csv` — proves the model holds up when AQI matters most, not just on average.
3. **Multi-horizon**: 24h / 48h / 72h ahead, compared side by side → `reports/model_comparison_by_horizon.csv` — shows how accuracy degrades further into the forecast.

All cross-validation during tuning uses `TimeSeriesSplit`, never shuffled k-fold (shuffling would leak future data into training).

**Models — all 7 wired in and actually trained:**
Persistence baseline, **Prophet** (classical statistical — added specifically to cover the brief's "statistical modelling" end of the spectrum), Ridge, Random Forest, XGBoost, LightGBM (all four tuned via **Optuna** + `TimeSeriesSplit`), plus LSTM and GRU (Keras, windowed sequence input, early stopping). Deliberately ordered statistical → deep learning to match the brief's phrasing. All 7 appear in `reports/model_comparison.csv`. **Only a tabular model** (Ridge/RF/XGBoost/LightGBM) is ever selected for the Hopsworks Model Registry / FastAPI serving path — Prophet needs a `ds` (date) column, LSTM/GRU need a windowed 3D input, both incompatible with the single-row `/predict` endpoint, so they're compared but not served. This is a deliberate, documented scope boundary (see the docstring at the top of `training_pipeline.py`), not an oversight.

**Framework choice — TensorFlow (Keras)** for both LSTM and GRU, not PyTorch: fewer moving parts, `model.fit()`/`model.save()` need less custom code to debug under a deadline, and both are secondary/comparison models here — the tree-based models are expected to win.

**Tools**: VS Code (primary IDE, all `.py`/Docker/YAML files) + Jupyter via VS Code's Jupyter extension (for `01_eda.ipynb`, the only notebook in this project — see `notebooks/README.md` for why) + Google Colab only as an optional fallback if local LSTM/GRU training feels slow — not needed for anything else, since the dataset (tens of thousands of hourly rows) trains fine on CPU.

## Tech Stack

- **Data**: OpenWeather Air Pollution API (pollution/AQI), Open-Meteo (weather)
- **Feature Store / Model Registry**: Hopsworks (serverless, free tier)
- **Modeling**: Prophet (statistical), scikit-learn (Ridge, Random Forest), XGBoost/LightGBM, TensorFlow/Keras (LSTM, GRU)
- **Explainability**: SHAP
- **Orchestration**: GitHub Actions
- **Serving**: FastAPI
- **Dashboard**: Streamlit
- **Containerization**: Docker + Docker Compose
- **Deployment**: Hugging Face Spaces / Streamlit Community Cloud

## Repository structure

```
aqi-predictor/
├── .github/workflows/       # CI/CD automation (hourly features, daily training)
├── src/
│   ├── config.py              # env vars, shared constants (coords, start date, smog months)
│   ├── feature_pipeline.py    # hourly run + backfill (OpenWeather + Open-Meteo, merged)
│   ├── training_pipeline.py   # season-aligned split, tuning, stratified + horizon eval, SHAP
│   ├── train_prophet.py       # classical statistical model (univariate, no leakage)
│   ├── train_lstm.py          # separate LSTM module (windowed 3D input)
│   ├── train_gru.py           # separate GRU module (windowed 3D input)
│   └── utils/
│       ├── data_fetch.py         # both APIs, UTC-aligned, + merge helper
│       ├── aqi_calculation.py    # real EPA AQI from PM2.5 (not OpenWeather's 1-5 field)
│       ├── data_validation.py    # required cols, ranges, nulls, duplicates
│       ├── optuna_tuning.py      # Optuna + TimeSeriesSplit tuning for all 4 tabular models
│       ├── feature_engineering.py # time/lag/rolling/season features, targets
│       └── hopsworks_utils.py    # shared connection + feature view setup
├── api/                     # FastAPI inference service
├── app/                     # Streamlit dashboard (multipage: EDA, Forecast, Alerts, Model Performance)
├── notebooks/                # 01_eda.ipynb only — see notebooks/README.md
├── tests/                    # unit tests (pytest)
├── reports/                  # SHAP plots, comparison CSVs, final_report.md
├── docker-compose.yml         # runs api + app together
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup — step by step

1. Clone the repo, open in VS Code, create a virtual environment, `pip install -r requirements.txt`.
2. Copy `.env.example` → `.env`, fill in `OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME` (Open-Meteo needs no key).
3. **Raw snapshot (Step 1 — before EDA / FE):** `python -m src.feature_pipeline raw-snapshot`  
   Fetches full history from both APIs, validates, and writes `data/raw/aqi_raw_merged.parquet`. No feature engineering, no Hopsworks.
4. Run `notebooks/01_eda.ipynb` on that raw parquet. Review the **Findings for FE** section before changing feature engineering.
5. Smoke-test hourly path (after FE is approved): `python -m src.feature_pipeline` — confirms both APIs + Hopsworks. Also upserts the raw snapshot.
6. Push engineered features to Hopsworks FG **v4** (from the local raw snapshot, no re-fetch):  
   `python -m src.feature_pipeline push-features`  
   (set `FEATURE_GROUP_VERSION=4` in `.env` if needed). Leave v2/v3 untouched as rollback.
7. Create the Hopsworks Feature View (one-time): `python -m src.utils.hopsworks_utils`.
8. Run `python -m src.training_pipeline` — trains all 7 models **per horizon** (24/48/72), scores vs persistence, registers `aqi_forecaster_{24,48,72}h` only when a model beats baseline on RMSE+MAE+R². Check `reports/`.
9. Add the same keys as GitHub **Repository Secrets**; push `.github/workflows/` and confirm both scheduled workflows succeed under the Actions tab.
10. Run locally: `uvicorn api.main:app --reload` + `streamlit run app/Home.py` — `/predict` returns day 1/2/3.
11. `docker compose up --build` — same check, containerized.
12. Deploy to Hugging Face Spaces (or Streamlit Community Cloud), add the live link below, write `reports/final_report.md`, submit the repo link.

## Live Demo

_(Add your deployed link here once deployed)_

## Report

See `reports/final_report.md` for EDA findings (including the smog vs. normal season comparison), feature engineering rationale, the full model comparison (overall, stratified, and per-horizon), and SHAP analysis.
