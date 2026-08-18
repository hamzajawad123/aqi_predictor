"""
Feature engineering for AQI forecasting.
Turns raw (timestamp, pollutants, weather) rows into the model-ready feature set.

IMPORTANT: this same function must be used by BOTH the feature pipeline (writing to
the feature store) and, conceptually, understood by the training pipeline (reading
from it) — so feature definitions never drift between training and serving.

EDA-justified choices (see notebooks/01_eda.ipynb Findings for FE):
- log1p on right-skewed pollutants
- lags {1,3,6,24,168} and rolling {3,6,24} from ACF/PACF
- delta targets to address multi-year downward AQI trend
"""
import numpy as np
import pandas as pd

from src import config

# Right-skewed pollutants from raw EDA (skew > 1 for all of these)
LOG_POLLUTANT_COLS = ("co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3")
AQI_LAGS = (1, 3, 6, 24, 168)
AQI_ROLL_WINDOWS = (3, 6, 24)
# Short outages get interpolated; anything longer stays NaN and those rows are
# dropped rather than invented. 6h is well under the shortest lag/target span.
MAX_INTERPOLATE_HOURS = 6


def to_hourly_grid(df: pd.DataFrame, ts_col: str = "timestamp",
                   interpolate_limit: int = MAX_INTERPOLATE_HOURS) -> pd.DataFrame:
    """
    Reindex onto a strict hourly grid before any shifting happens.

    Every lag, rolling window, change rate and target below shifts by ROW
    POSITION, which only equals hours when no hour is missing. The raw history
    has ~3,700 missing hours in 415 outages, so on the gappy frame shift(-24)
    actually spanned 24h for only 87% of rows and up to 264h for the rest —
    i.e. those rows were trained against the wrong future. Reindexing first
    makes position and time mean the same thing again; gaps longer than
    interpolate_limit stay NaN and get dropped by build_feature_set.
    """
    df = df.sort_values(ts_col).drop_duplicates(subset=ts_col)
    grid = pd.date_range(df[ts_col].min(), df[ts_col].max(), freq="h")
    out = df.set_index(ts_col).reindex(grid).rename_axis(ts_col).reset_index()

    if not interpolate_limit:
        return out

    # Fill only outages that are short END TO END. Pandas' own `limit` caps
    # consecutive fills instead, which would patch the first 6 hours of a
    # 20-hour outage by interpolating toward a reading 20 hours away — that
    # invents a trajectory rather than bridging a blip.
    inserted = ~out[ts_col].isin(set(df[ts_col]))
    run_id = (inserted != inserted.shift()).cumsum()
    run_length = out.groupby(run_id)[ts_col].transform("size")
    fillable = inserted & (run_length <= interpolate_limit)

    numeric = out.select_dtypes("number").columns
    interpolated = out[numeric].interpolate(limit_area="inside")
    out.loc[fillable, numeric] = interpolated.loc[fillable, numeric]
    return out


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
    df = df.copy()
    df["is_smog_season"] = df["month"].isin(config.SMOG_SEASON_MONTHS).astype(int)
    return df


def add_log_pollutants(df: pd.DataFrame,
                       cols: tuple[str, ...] = LOG_POLLUTANT_COLS) -> pd.DataFrame:
    """
    log1p transform for right-skewed pollutant concentrations (EDA skew >> 1).
    Replaces the raw column in-place so downstream interactions/lags stay simple.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def add_lag_features(df: pd.DataFrame, col: str = "aqi",
                      lags: tuple[int, ...] = AQI_LAGS) -> pd.DataFrame:
    """Lag features justified by ACF/PACF on raw AQI (1h, 3h, 6h, 24h, 168h)."""
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str = "aqi",
                          windows: tuple[int, ...] = AQI_ROLL_WINDOWS) -> pd.DataFrame:
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
                 horizons: tuple[int, ...] = None) -> pd.DataFrame:
    """
    Absolute future AQI at t+h (for metrics / Prophet) plus delta targets
    (aqi_target - aqi) used as the primary train target for tabular/RNN models.
    """
    horizons = horizons or config.TARGET_HORIZONS
    df = df.copy()
    for h in horizons:
        abs_col = f"{col}_target_{h}h"
        df[abs_col] = df[col].shift(-h)
        df[f"{col}_delta_{h}h"] = df[abs_col] - df[col]
    return df


def build_feature_set(raw_df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Full pipeline: raw dataframe -> model-ready features + absolute/delta targets.

    is_training=True drops any row with a NaN anywhere, so every row has both
    complete lags and a known future. is_training=False keeps rows whose
    targets aren't knowable yet, which is the only way the most recent hours
    survive: aqi_target_72h needs 72 hours that haven't happened, so requiring
    it discards exactly the rows inference needs.

    Target columns are built in BOTH modes (as NaN when the future is unknown)
    so the returned schema never depends on the caller — the feature group is
    created from this frame, and a mode-dependent column set would either fail
    schema validation or silently fork the stored table.
    """
    # aqi, humidity and wind_deg arrive as whole numbers, and the feature store
    # schema types them as bigint. Interpolating across a gap makes them
    # fractional, so round straight away (before lags/targets are derived from
    # them, so every derived column agrees) and restore the dtype at the end,
    # once the NaN rows an integer column can't hold are gone.
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
    # Rows at the start (no lag history) and end (no future target) become NaN —
    # drop them here rather than silently letting the model choke on them later.
    if is_training:
        df = df.dropna().reset_index(drop=True)
    else:
        feature_cols = [
            c for c in df.columns
            if not c.startswith(("aqi_target_", "aqi_delta_"))
        ]
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
    # Safe in both modes: every integer column is a feature, and features are
    # required non-null above, so there's no NaN left for int64 to choke on.
    for col in integer_cols:
        df[col] = df[col].astype("int64")
    return df
