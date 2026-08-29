"""Build model features from raw pollution + weather rows."""
import numpy as np
import pandas as pd

from src import config

LOG_POLLUTANT_COLS = ("co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3")
AQI_LAGS = (1, 3, 6, 24, 168)
AQI_ROLL_WINDOWS = (3, 6, 24)
# Fill gaps of 6 hours or less. Longer gaps stay empty and get dropped.
MAX_INTERPOLATE_HOURS = 6


def to_hourly_grid(df: pd.DataFrame, ts_col: str = "timestamp",
                   interpolate_limit: int = MAX_INTERPOLATE_HOURS) -> pd.DataFrame:
    """Make one row per hour so lags mean real hours, not just row counts."""
    df = df.sort_values(ts_col).drop_duplicates(subset=ts_col)
    grid = pd.date_range(df[ts_col].min(), df[ts_col].max(), freq="h")
    out = df.set_index(ts_col).reindex(grid).rename_axis(ts_col).reset_index()

    if not interpolate_limit:
        return out

    # Only fill short gaps. Do not fill the start of a long outage.
    inserted = ~out[ts_col].isin(set(df[ts_col]))
    run_id = (inserted != inserted.shift()).cumsum()
    run_length = out.groupby(run_id)[ts_col].transform("size")
    fillable = inserted & (run_length <= interpolate_limit)

    numeric = out.select_dtypes("number").columns
    interpolated = out[numeric].interpolate(limit_area="inside")
    out.loc[fillable, numeric] = interpolated.loc[fillable, numeric]
    return out


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Hour and month as sin/cos so 23 and 0 sit next to each other."""
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
    """1 in Oct–Jan, else 0."""
    df = df.copy()
    df["is_smog_season"] = df["month"].isin(config.SMOG_SEASON_MONTHS).astype(int)
    return df


def add_log_pollutants(df: pd.DataFrame,
                       cols: tuple[str, ...] = LOG_POLLUTANT_COLS) -> pd.DataFrame:
    """log1p on skewed pollutants. Replaces the raw column."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def add_lag_features(df: pd.DataFrame, col: str = "aqi",
                      lags: tuple[int, ...] = AQI_LAGS) -> pd.DataFrame:
    """Past AQI at 1, 3, 6, 24 and 168 hours."""
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str = "aqi",
                          windows: tuple[int, ...] = AQI_ROLL_WINDOWS) -> pd.DataFrame:
    """Rolling mean / std / min / max of AQI."""
    df = df.copy()
    for w in windows:
        df[f"{col}_roll_mean_{w}h"] = df[col].rolling(w).mean()
        df[f"{col}_roll_std_{w}h"] = df[col].rolling(w).std()
        df[f"{col}_roll_min_{w}h"] = df[col].rolling(w).min()
        df[f"{col}_roll_max_{w}h"] = df[col].rolling(w).max()
    return df


def add_change_rate(df: pd.DataFrame, col: str = "aqi") -> pd.DataFrame:
    """How much AQI moved in the last 1h and 24h."""
    df = df.copy()
    df[f"{col}_change_rate_1h"] = df[col].diff(1)
    df[f"{col}_change_rate_24h"] = df[col].diff(24)
    return df


def add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Wind × PM2.5 and humidity × PM2.5."""
    df = df.copy()
    if {"wind_speed", "pm2_5"}.issubset(df.columns):
        df["wind_pollutant_interaction"] = df["wind_speed"] * df["pm2_5"]
    if {"humidity", "pm2_5"}.issubset(df.columns):
        df["humidity_pollutant_interaction"] = df["humidity"] * df["pm2_5"]
    return df


def add_targets(df: pd.DataFrame, col: str = "aqi",
                 horizons: tuple[int, ...] = None) -> pd.DataFrame:
    """Future AQI and the change from now, for 24 / 48 / 72 hours."""
    horizons = horizons or config.TARGET_HORIZONS
    df = df.copy()
    for h in horizons:
        abs_col = f"{col}_target_{h}h"
        df[abs_col] = df[col].shift(-h)
        df[f"{col}_delta_{h}h"] = df[abs_col] - df[col]
    return df


def build_feature_set(raw_df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Raw rows to features + targets. Training drops rows with missing future AQI."""
    # Round whole-number cols so Hopsworks can store them as bigint.
    integer_cols = [c for c in raw_df.columns if pd.api.types.is_integer_dtype(raw_df[c])]

    df = to_hourly_grid(raw_df)
    for col in integer_cols:
        df[col] = df[col].round()
    df = add_time_features(df)
    df = add_season_flag(df)
    df = add_log_pollutants(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_change_rate(df)
    df = add_weather_interactions(df)
    df = add_targets(df)
    # First and last rows lack lags or future AQI.
    if is_training:
        df = df.dropna().reset_index(drop=True)
    else:
        feature_cols = [
            c for c in df.columns
            if not c.startswith(("aqi_target_", "aqi_delta_"))
        ]
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
    for col in integer_cols:
        df[col] = df[col].astype("int64")
    return df
