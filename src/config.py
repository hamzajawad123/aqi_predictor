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
AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN")
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
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", 1))
FEATURE_VIEW_NAME = os.getenv("FEATURE_VIEW_NAME", "aqi_feature_view")
MODEL_NAME = os.getenv("MODEL_NAME", "aqi_forecaster")

# --- Forecast config ---
FORECAST_HORIZON_HOURS = 72  # 3 days ahead
TARGET_HORIZONS = (24, 48, 72)  # all three evaluated separately in the report

# --- Data alignment ---
# OpenWeather's air pollution history starts 27 Nov 2020. Open-Meteo's weather
# archive goes back much further, but we deliberately start BOTH sources on
# this exact same date so the two dataframes align from row zero (see the
# "start both datasets from the same day" discussion).
DATA_START_DATE = os.getenv("DATA_START_DATE", "2020-11-27")

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
