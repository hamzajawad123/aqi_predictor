"""Fetch pollution from OpenWeather and weather from Open-Meteo. All times are UTC."""
import requests
import pandas as pd
from src import config

# OpenWeather — pollution only
AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
AIR_POLLUTION_FORECAST_URL = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
AIR_POLLUTION_HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

# Open-Meteo — weather only
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

def fetch_current_air_pollution() -> pd.DataFrame:
    """Live pollution for the city."""
    params = {
        "lat": config.LATITUDE,
        "lon": config.LONGITUDE,
        "appid": config.OPENWEATHER_API_KEY,
    }
    resp = requests.get(AIR_POLLUTION_URL, params=params, timeout=15)
    resp.raise_for_status()
    return _pollution_json_to_df(resp.json())


def fetch_air_pollution_forecast() -> pd.DataFrame:
    """OpenWeather 4-day pollution forecast. Not used in training."""
    params = {
        "lat": config.LATITUDE,
        "lon": config.LONGITUDE,
        "appid": config.OPENWEATHER_API_KEY,
    }
    resp = requests.get(AIR_POLLUTION_FORECAST_URL, params=params, timeout=15)
    resp.raise_for_status()
    return _pollution_json_to_df(resp.json())


def fetch_historical_air_pollution(start_unix: int, end_unix: int) -> pd.DataFrame:
    """Past pollution between two UTC unix times. Free from 27 Nov 2020."""
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
    """OpenWeather JSON to a table. aqi is EPA from PM2.5, not the 1–5 score."""
    from src.utils.aqi_calculation import pm25_to_aqi

    rows = []
    for item in payload.get("list", []):
        row = {
            "timestamp": item["dt"],
            "openweather_aqi_category": item["main"]["aqi"],  # 1–5, not the target
        }
        row.update(item["components"])
        row["aqi"] = pm25_to_aqi(row["pm2_5"])
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_localize(None)
    return df


# Weather


def fetch_openmeteo_current_weather() -> pd.DataFrame:
    """Live weather. timezone=UTC so it lines up with OpenWeather."""
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
    """Weather for the last few days. Use this, not the archive, for yesterday."""
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
    """Archive weather. Dates as YYYY-MM-DD. Always UTC."""
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
    """Open-Meteo air quality. Not used in the main pipeline."""
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


# Merge


def merge_pollution_and_weather(pollution_df: pd.DataFrame,
                                 weather_df: pd.DataFrame) -> pd.DataFrame:
    """Inner join on timestamp. Both sides are already UTC hourly."""
    if pollution_df.empty or weather_df.empty:
        return pd.DataFrame()
    merged = pd.merge(pollution_df, weather_df, on="timestamp", how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)
