"""
Repairs the historical data currently sitting in your `aqi_features` Hopsworks
feature group (exported as features.csv). This version is aligned EXACTLY to
src/utils/feature_engineering.py's build_feature_set() -- same lag windows,
same rolling windows, same smog-season months from config.py, same
wind/humidity interaction formulas -- so the repaired data is indistinguishable
from what the (now-fixed) pipeline would have produced with no gaps.

Root cause being fixed: 407 gaps / 3,617 missing hours in the stored data.
Because build_feature_set() computes lags/rolling stats/targets with plain
.shift()/.rolling() (row-based), any gap in the timestamp index made those
features silently wrong across the gap (confirmed directly: aqi_lag_1h right
after a 4h gap held a 4h-old value, not a 1h-old one). Filling every missing
hour first, THEN recomputing every derived column from scratch, is the only
way to fix that retroactively.

Usage:
    python fix_features_pipeline_v2.py
Produces features_fixed.csv, ready to insert into a (new version of the)
aqi_features feature group. See the numbered steps in chat for the Hopsworks
side of this.
"""
import numpy as np
import pandas as pd

IN_PATH = "features.csv"
OUT_PATH = "features_fixed.csv"

# From src/config.py -- keep these in sync if you ever change config.py
SMOG_SEASON_MONTHS = (10, 11, 12, 1)
LAGS = (1, 3, 6, 24, 168)
ROLLING_WINDOWS = (3, 6, 24)
TARGET_HORIZONS = (24, 48, 72)

RAW_INTERP_COLS = [
    "openweather_aqi_category", "co", "no", "no2", "o3", "so2", "pm2_5",
    "pm10", "nh3", "aqi", "temperature", "humidity", "wind_speed",
    "wind_deg", "pressure",
]

# ---- 1. Load, sort, dedup ----
df = pd.read_csv(IN_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
df = df.set_index("timestamp")

# ---- 2. Drop the 4 contradictory rows (aqi=0 but category says polluted) ----
before = len(df)
df = df[~((df["aqi"] == 0) & (df["openweather_aqi_category"] >= 4))]
print(f"Dropped {before - len(df)} contradictory rows.")

# ---- 3. Fill every missing hour with a continuous hourly index ----
df = df.resample("h").asfreq()
n_filled = int(df["aqi"].isna().sum())
df[RAW_INTERP_COLS] = df[RAW_INTERP_COLS].interpolate(method="time")
print(f"Filled {n_filled} missing hours via time interpolation.")

df["aqi"] = df["aqi"].round().astype(int)
df["openweather_aqi_category"] = df["openweather_aqi_category"].round().clip(1, 5).astype(int)
df["humidity"] = df["humidity"].round().astype(int)
df["wind_deg"] = df["wind_deg"].round().astype(int)

df = df.reset_index()

# ---- 4. Recompute everything downstream from the fixed base columns,
#          mirroring build_feature_set()'s exact steps and order ----

# add_time_features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# add_season_flag
df["is_smog_season"] = df["month"].isin(SMOG_SEASON_MONTHS).astype(int)

# add_lag_features
for lag in LAGS:
    df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)

# add_rolling_features
for w in ROLLING_WINDOWS:
    df[f"aqi_roll_mean_{w}h"] = df["aqi"].rolling(w).mean()
    df[f"aqi_roll_std_{w}h"] = df["aqi"].rolling(w).std()
    df[f"aqi_roll_min_{w}h"] = df["aqi"].rolling(w).min()
    df[f"aqi_roll_max_{w}h"] = df["aqi"].rolling(w).max()

# add_change_rate
df["aqi_change_rate_1h"] = df["aqi"].diff(1)
df["aqi_change_rate_24h"] = df["aqi"].diff(24)

# add_weather_interactions -- exact formula confirmed from feature_engineering.py
df["wind_pollutant_interaction"] = df["wind_speed"] * df["pm2_5"]
df["humidity_pollutant_interaction"] = df["humidity"] * df["pm2_5"]

# add_targets -- includes the new 72h target
for h in TARGET_HORIZONS:
    df[f"aqi_target_{h}h"] = df["aqi"].shift(-h)

# ---- 5. dropna(), exactly like build_feature_set() does at the end ----
before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"Dropped {before - len(df)} rows with incomplete lag/target history.")

# ---- sanity checks ----
assert df["timestamp"].is_monotonic_increasing
assert (df["timestamp"].diff().dropna() == pd.Timedelta("1h")).all(), "still has gaps!"
assert df.isna().sum().sum() == 0
assert "aqi_target_72h" in df.columns

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(df)} rows x {len(df.columns)} cols -> {OUT_PATH}")
