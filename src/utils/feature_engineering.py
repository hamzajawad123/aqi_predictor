"""
Feature engineering for AQI forecasting.
Turns raw (timestamp, pollutants, weather) rows into the model-ready feature set.

IMPORTANT: this same function must be used by BOTH the feature pipeline (writing to
the feature store) and, conceptually, understood by the training pipeline (reading
from it) — so feature definitions never drift between training and serving.
"""
import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Cyclical hour/day/month encodings, so e.g. hour=23 and hour=0 are 'close'."""
    df = df.copy()
    df["hour"] = df[ts_col].dt.hour
    df["day_of_week"] = df[ts_col].dt.dayofweek
    df["month"] = df[ts_col].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_season_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lahore smog-season flag (Oct-Jan). Used both as a model feature and,
    unchanged, as the stratification key for smog-vs-normal evaluation in
    training_pipeline.py — keeping one definition avoids the two ever
    silently drifting apart.
    """
    from src import config
    df = df.copy()
    df["is_smog_season"] = df["month"].isin(config.SMOG_SEASON_MONTHS).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, col: str = "aqi",
                      lags=(1, 3, 6, 24, 168)) -> pd.DataFrame:
    """Lag features in hours: 1h, 3h, 6h, 24h (1 day), 168h (1 week)."""
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str = "aqi",
                          windows=(3, 6, 24)) -> pd.DataFrame:
    """Rolling mean/std/min/max — captures short-term trend and volatility."""
    df = df.copy()
    for w in windows:
        df[f"{col}_roll_mean_{w}h"] = df[col].rolling(w).mean()
        df[f"{col}_roll_std_{w}h"] = df[col].rolling(w).std()
        df[f"{col}_roll_min_{w}h"] = df[col].rolling(w).min()
        df[f"{col}_roll_max_{w}h"] = df[col].rolling(w).max()
    return df


def add_change_rate(df: pd.DataFrame, col: str = "aqi") -> pd.DataFrame:
    """AQI change rate (explicitly required by the project brief)."""
    df = df.copy()
    df[f"{col}_change_rate_1h"] = df[col].diff(1)
    df[f"{col}_change_rate_24h"] = df[col].diff(24)
    return df


def add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lahore-specific interactions: wind disperses pollution, humidity affects
    secondary particle formation. Requires wind_speed/humidity/pm2_5 columns.
    """
    df = df.copy()
    if {"wind_speed", "pm2_5"}.issubset(df.columns):
        df["wind_pollutant_interaction"] = df["wind_speed"] * df["pm2_5"]
    if {"humidity", "pm2_5"}.issubset(df.columns):
        df["humidity_pollutant_interaction"] = df["humidity"] * df["pm2_5"]
    return df


def add_targets(df: pd.DataFrame, col: str = "aqi",
                 horizons=(24, 48, 72)) -> pd.DataFrame:
    """
    Targets for a 3-day-ahead forecast: AQI at t+24h, t+48h, t+72h.
    Using multiple horizons lets you train either 3 separate models or a
    multi-output model, depending on what your Step-3 experiments favor.
    """
    df = df.copy()
    for h in horizons:
        df[f"{col}_target_{h}h"] = df[col].shift(-h)
    return df


def build_feature_set(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: raw dataframe -> model-ready features + targets."""
    df = raw_df.sort_values("timestamp").reset_index(drop=True)
    df = add_time_features(df)
    df = add_season_flag(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_change_rate(df)
    df = add_weather_interactions(df)
    df = add_targets(df)
    # Rows at the start (no lag history) and end (no future target) become NaN —
    # drop them here rather than silently letting the model choke on them later.
    df = df.dropna().reset_index(drop=True)
    return df
