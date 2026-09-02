"""Settings loaded from .env. Import this instead of calling os.getenv everywhere."""
import os
from dotenv import load_dotenv

load_dotenv()

# Keys
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
# Pin the host so the SDK does not pick the wrong cluster.
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

# City
CITY_NAME = os.getenv("CITY_NAME", "Lahore")
LATITUDE = float(os.getenv("LATITUDE", 31.5497))
LONGITUDE = float(os.getenv("LONGITUDE", 74.3436))

# Hopsworks names — keep the same in feature, training and serving
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME", "aqi_features")
# v4 uses a real hourly grid. Older versions had gappy lags.
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", 4))
FEATURE_VIEW_NAME = os.getenv("FEATURE_VIEW_NAME", "aqi_feature_view")
MODEL_NAME = os.getenv("MODEL_NAME", "aqi_forecaster")


def model_name_for_horizon(horizon_hours: int) -> str:
    """Model name for one horizon, e.g. aqi_forecaster_24h."""
    return f"{MODEL_NAME}_{int(horizon_hours)}h"

FORECAST_HORIZON_HOURS = 72  # 3 days
TARGET_HORIZONS = (24, 48, 72)

# OpenWeather history starts on this day, so weather starts here too.
DATA_START_DATE = os.getenv("DATA_START_DATE", "2020-11-27")

# OpenWeather AQI jumps became much smaller after this date.
REGIME_BREAK_DATE = "2025-04-04"

# Skip rows before this date. Set TRAIN_START_DATE="" to use all history.
TRAIN_START_DATE = os.getenv("TRAIN_START_DATE", REGIME_BREAK_DATE) or None

# Pull predicted change toward 0. 0 = no change, 1 = full model.
USE_DELTA_SHRINKAGE = os.getenv("USE_DELTA_SHRINKAGE", "true").lower() == "true"

# Local merge of pollution + weather. EDA reads this file.
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/aqi_raw_merged.parquet")

# Oct–Jan
SMOG_SEASON_MONTHS = (10, 11, 12, 1)


def validate_config(*, require_openweather: bool = True):
    """Training only needs Hopsworks. Fetch/hourly/backfill also need OpenWeather."""
    required = {
        "HOPSWORKS_API_KEY": HOPSWORKS_API_KEY,
        "HOPSWORKS_PROJECT_NAME": HOPSWORKS_PROJECT_NAME,
    }
    if require_openweather:
        required["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}. "
            f"Copy .env.example to .env and fill in real values."
        )
