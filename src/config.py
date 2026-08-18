"""
Central configuration for the AQI Predictor project.
Loads environment variables once so every pipeline/script imports from here
instead of scattering os.getenv() calls everywhere.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys / secrets ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
# Without an explicit host, the SDK has to guess/resolve which Hopsworks
# cluster to connect to — and that guess isn't always stable between runs
# (seen firsthand: one run correctly used eu-west.cloud.hopsworks.ai, the
# very next run tried c.app.hopsworks.ai instead, a hostname that doesn't
# even resolve in DNS). Pinning it explicitly removes that ambiguity.
# Check your Hopsworks dashboard's URL to confirm which region your project
# is actually on if this default isn't right for your account.
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

# --- Location ---
CITY_NAME = os.getenv("CITY_NAME", "Lahore")
LATITUDE = float(os.getenv("LATITUDE", 31.5497))
LONGITUDE = float(os.getenv("LONGITUDE", 74.3436))

# --- Hopsworks object names (keep identical across feature/training/inference pipelines) ---
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME", "aqi_features")
# v1/v2 = prior engineered tables (rollback). v3 = post-EDA FE with delta
# targets, but built on the gappy frame so ~13% of its lags/targets span the
# wrong number of hours. v4 = same features on a strict hourly grid.
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", 4))
FEATURE_VIEW_NAME = os.getenv("FEATURE_VIEW_NAME", "aqi_feature_view")
# Base name; per-horizon registry entries are f"{MODEL_NAME}_{h}h"
MODEL_NAME = os.getenv("MODEL_NAME", "aqi_forecaster")


def model_name_for_horizon(horizon_hours: int) -> str:
    """Registry name for a single day-ahead horizon (24 / 48 / 72)."""
    return f"{MODEL_NAME}_{int(horizon_hours)}h"

# --- Forecast config ---
FORECAST_HORIZON_HOURS = 72  # 3 days ahead
TARGET_HORIZONS = (24, 48, 72)  # all three evaluated separately in the report

# --- Data alignment ---
# OpenWeather's air pollution history starts 27 Nov 2020. Open-Meteo's weather
# archive goes back much further, but we deliberately start BOTH sources on
# this exact same date so the two dataframes align from row zero (see the
# "start both datasets from the same day" discussion).
DATA_START_DATE = os.getenv("DATA_START_DATE", "2020-11-27")

# --- Training window ---
# OpenWeather's air pollution history changes character on this date: mean
# hourly |AQI change| collapses from ~46 to ~4.5 and never recovers (measured
# per month across the whole archive), and the same break shows up in pm2_5.
# Everything before it is a different data-generating process, so training
# across the break teaches models a volatility that no longer exists — that is
# what made every model lose to persistence on the 2025-26 test period.
REGIME_BREAK_DATE = "2025-04-04"

# Rows before this date are excluded from training. Defaults to the regime
# break; set TRAIN_START_DATE="" in .env to deliberately train on all history.
TRAIN_START_DATE = os.getenv("TRAIN_START_DATE", REGIME_BREAK_DATE) or None

# Shrink predicted deltas toward 0 by a factor fitted on validation
# (0 = persistence, 1 = raw model). See utils.evaluation.fit_delta_shrinkage.
USE_DELTA_SHRINKAGE = os.getenv("USE_DELTA_SHRINKAGE", "true").lower() == "true"

# --- Local raw snapshot (pre-feature-engineering merge of pollution + weather) ---
# Written by `python -m src.feature_pipeline raw-snapshot` and upserted by
# hourly/backfill runs. EDA reads this path — never the engineered feature group.
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/aqi_raw_merged.parquet")

# Lahore's smog season (used for stratified evaluation: smog vs. normal season)
SMOG_SEASON_MONTHS = (10, 11, 12, 1)

# --- Sanity check on import (fails fast instead of failing deep in a pipeline) ---
def validate_config():
    required = {
        "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
        "HOPSWORKS_API_KEY": HOPSWORKS_API_KEY,
        "HOPSWORKS_PROJECT_NAME": HOPSWORKS_PROJECT_NAME,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}. "
            f"Copy .env.example to .env and fill in real values."
        )
