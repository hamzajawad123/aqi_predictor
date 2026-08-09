"""
Raw data fetching from external APIs.

FINAL SOURCE DECISION (see project discussion):
- Pollution / AQI  -> OpenWeather Air Pollution API (current, forecast, historical).
  One source, always, for the actual prediction target — never mixed with
  another provider's pollution readings.
- Weather (temp/humidity/wind) -> Open-Meteo (current AND historical). One
  source, always, for weather too. Earlier drafts used OpenWeather's live
  weather endpoint for the hourly pipeline but Open-Meteo for backfill —
  that mismatch is exactly the train/serving skew a feature store exists to
  prevent, so weather now comes from Open-Meteo everywhere, hourly and
  historical alike.

All timestamps from both sources are normalized to UTC before being returned,
so merging them is a plain equi-join on `timestamp` with no manual offset
math needed.
"""
import requests
import pandas as pd
from src import config

# --- OpenWeather (pollution/AQI only) ---
AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
AIR_POLLUTION_FORECAST_URL = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
AIR_POLLUTION_HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

# --- Open-Meteo (weather only) ---
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_OPENMETEO_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,"
    "surface_pressure"
)
_OPENMETEO_RENAME = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_deg",
    "surface_pressure": "pressure",
}


# ---------------------------------------------------------------------------
# Pollution / AQI — OpenWeather
# ---------------------------------------------------------------------------

def fetch_current_air_pollution() -> pd.DataFrame:
    """Current AQI + pollutant concentrations for the configured city."""
    params = {
        "lat": config.LATITUDE,
        "lon": config.LONGITUDE,
        "appid": config.OPENWEATHER_API_KEY,
    }
    resp = requests.get(AIR_POLLUTION_URL, params=params, timeout=15)
    resp.raise_for_status()
    return _pollution_json_to_df(resp.json())


def fetch_air_pollution_forecast() -> pd.DataFrame:
    """4-day-ahead pollution forecast (hourly). Not used by the current
    training pipeline, but useful if you later want the model to compare
    against OpenWeather's own forecast as a baseline."""
    params = {
        "lat": config.LATITUDE,
        "lon": config.LONGITUDE,
        "appid": config.OPENWEATHER_API_KEY,
    }
    resp = requests.get(AIR_POLLUTION_FORECAST_URL, params=params, timeout=15)
    resp.raise_for_status()
    return _pollution_json_to_df(resp.json())


def fetch_historical_air_pollution(start_unix: int, end_unix: int) -> pd.DataFrame:
    """
    Historical AQI + pollutant data between two UNIX (UTC) timestamps.
    Free since 27 Nov 2020. Used by the backfill script.
    """
    params = {
        "lat": config.LATITUDE,
        "lon": config.LONGITUDE,
        "start": start_unix,
        "end": end_unix,
        "appid": config.OPENWEATHER_API_KEY,
    }
    resp = requests.get(AIR_POLLUTION_HISTORY_URL, params=params, timeout=15)
    resp.raise_for_status()
    return _pollution_json_to_df(resp.json())


def _pollution_json_to_df(payload: dict) -> pd.DataFrame:
    """
    Flatten OpenWeather's air pollution JSON response into a tidy dataframe.
    OpenWeather's `dt` field is already a UTC Unix timestamp.

    IMPORTANT: `aqi` here is the REAL US EPA AQI, computed from PM2.5 (see
    aqi_calculation.py) — NOT OpenWeather's own `main.aqi` field, which is
    only a coarse 1-5 category. That raw OpenWeather value is kept as
    `openweather_aqi_category` for reference/EDA only; it is never used as
    a model feature or target.
    """
    from src.utils.aqi_calculation import pm25_to_aqi

    rows = []
    for item in payload.get("list", []):
        row = {
            "timestamp": item["dt"],
            "openweather_aqi_category": item["main"]["aqi"],  # reference only, 1-5
        }
        row.update(item["components"])  # co, no, no2, o3, so2, pm2_5, pm10, nh3
        row["aqi"] = pm25_to_aqi(row["pm2_5"])  # the REAL continuous AQI, our actual target
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_localize(None)
    return df


# ---------------------------------------------------------------------------
# Weather — Open-Meteo (current + historical, same provider both places)
# ---------------------------------------------------------------------------

def fetch_openmeteo_current_weather() -> pd.DataFrame:
    """
    Live weather via Open-Meteo's forecast endpoint's `current` block.
    Returned as a one-row dataframe so it merges the same way historical
    weather does. `timezone=UTC` is set EXPLICITLY — Open-Meteo's `auto`
    option returns local time, which silently misaligns with OpenWeather's
    UTC timestamps if you forget to convert it back.
    """
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                   "wind_direction_10m,surface_pressure",
        "timezone": "UTC",
    }
    resp = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=15)
    resp.raise_for_status()
    current = resp.json()["current"]
    df = pd.DataFrame([current])
    df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    return df.rename(columns=_OPENMETEO_RENAME).drop(columns=["time", "interval"], errors="ignore")


def fetch_openmeteo_recent_weather(past_days: int = 9) -> pd.DataFrame:
    """
    Recent weather (last `past_days` days, hourly) via Open-Meteo's FORECAST
    endpoint's `past_days` param -- NOT the archive endpoint, which lags
    ~5 days behind for reanalysis QC and would return nothing for "yesterday".
    Used by the hourly pipeline to get enough lookback history to compute
    lag/rolling features for the current hour (see feature_pipeline.py).
    """
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": _OPENMETEO_HOURLY_VARS,
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "UTC",
    }
    resp = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    return df.rename(columns=_OPENMETEO_RENAME).drop(columns=["time"])


def fetch_openmeteo_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Historical weather via Open-Meteo's archive endpoint.
    `timezone=UTC` explicit (see note above — `auto` was a real bug in an
    earlier draft of this function, since it silently returned local
    Asia/Karachi time instead of UTC, misaligning every row against
    OpenWeather's UTC pollution timestamps by 5 hours).
    start_date/end_date format: 'YYYY-MM-DD'.
    """
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": _OPENMETEO_HOURLY_VARS,
        "timezone": "UTC",
    }
    resp = requests.get(OPENMETEO_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    return df.rename(columns=_OPENMETEO_RENAME).drop(columns=["time"])


def fetch_openmeteo_air_quality() -> pd.DataFrame:
    """
    OPTIONAL cross-check only (not used in the main pipeline): Open-Meteo's
    own Air Quality API (CAMS-based), useful for validating OpenWeather's AQI
    numbers in your report, or as a fallback if OpenWeather's quota runs out.
    Do NOT mix this into the same feature column as OpenWeather's AQI —
    keep it as a separate, clearly-labelled comparison column if you use it.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide",
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    return df.drop(columns=["time"])


# ---------------------------------------------------------------------------
# Merge helper — combine pollution (OpenWeather) + weather (Open-Meteo)
# ---------------------------------------------------------------------------

def merge_pollution_and_weather(pollution_df: pd.DataFrame,
                                 weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Both dataframes are UTC and hourly by construction, so this is a plain
    inner join on `timestamp` — no timezone math needed at merge time
    because it was already handled when each was fetched.
    """
    if pollution_df.empty or weather_df.empty:
        return pd.DataFrame()
    merged = pd.merge(pollution_df, weather_df, on="timestamp", how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)
