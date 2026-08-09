"""
Feature Pipeline
================
Runs HOURLY via .github/workflows/feature_pipeline.yml

1. Fetch raw pollutant data (OpenWeather) + weather data (Open-Meteo).
2. Merge them on their shared UTC timestamp.
3. Compute features (time-based, lags, rolling stats, AQI change rate, season flag).
4. Write the resulting row(s) to the Hopsworks Feature Store.

Also usable directly for BACKFILL: fetches the full historical range in one
run to seed training data before automation starts. See backfill_historical().

IMPORTANT: pollution always comes from OpenWeather and weather always comes
from Open-Meteo, in BOTH this hourly path and the backfill path. Using
different sources for the same variable between backfill and live collection
is exactly the train/serving skew a feature store exists to prevent — so
don't swap either source in just one of the two functions below.
"""
import sys
import pandas as pd

from src import config
from src.utils.data_fetch import (
    fetch_historical_air_pollution,
    fetch_openmeteo_recent_weather,
    fetch_openmeteo_historical_weather,
    merge_pollution_and_weather,
)
from src.utils.feature_engineering import build_feature_set
from src.utils.hopsworks_utils import get_feature_store, get_or_create_feature_group
from src.utils.data_validation import validate_raw_data

# aqi_lag_168h needs 168h of history behind the current hour. Padded well
# above that so a handful of missed/late hours upstream doesn't reintroduce
# the same "not enough rows to compute lags" problem this fixes.
LOOKBACK_HOURS = 200


def run_hourly_feature_pipeline():
    """
    IMPORTANT FIX: this used to fetch only the CURRENT hour (1 row) from each
    source, then call build_feature_set() on that single row. Lag/rolling
    features need real history to compute, so on a 1-row input they were all
    NaN -- and build_feature_set()'s trailing dropna() then dropped that one
    row entirely. Net effect: fg.insert() ran on an EMPTY dataframe every
    single hour, silently inserting 0 rows every run since automation started.
    This is why the feature store has 407 gaps / 3,617 missing hours -- the
    only rows that ever made it in came from manual backfill_historical() runs.

    Fix: fetch a LOOKBACK_HOURS window (enough for aqi_lag_168h to have real
    data), run build_feature_set on the whole window so lags/rolling are
    correctly computed, then insert only the rows for hours not already in
    the feature store (skip-if-exists, so this can also self-heal small gaps
    from a missed run without duplicating what's already there).
    """
    config.validate_config()

    now = pd.Timestamp.utcnow().floor("h")
    lookback_start = now - pd.Timedelta(hours=LOOKBACK_HOURS)

    pollution_df = fetch_historical_air_pollution(
        int(lookback_start.timestamp()), int(now.timestamp())
    )
    weather_df = fetch_openmeteo_recent_weather(
        past_days=(LOOKBACK_HOURS // 24) + 2
    )

    print(f"[feature_pipeline] Debug — pollution_df rows: {len(pollution_df)}, "
          f"weather_df rows: {len(weather_df)}")

    if pollution_df.empty or weather_df.empty:
        print("[feature_pipeline] One of the two sources returned no data this "
              "run (see debug output above) — skipping.")
        return

    merged_df = merge_pollution_and_weather(pollution_df, weather_df)
    # Keep only the lookback window -- Open-Meteo's forecast_days=1 can hand
    # back a couple of hours past `now`; those aren't real observations yet.
    merged_df = merged_df[merged_df["timestamp"] <= now]

    if merged_df.empty:
        print("[feature_pipeline] No overlapping timestamps between the two "
              "sources this run — skipping (will retry next hour).")
        return

    merged_df = validate_raw_data(merged_df)
    if merged_df.empty:
        print("[feature_pipeline] All rows failed validation this run — skipping "
              "(will retry next hour).")
        return

    features_df = build_feature_set(merged_df)
    if features_df.empty:
        print("[feature_pipeline] Not enough history in this window yet to "
              "compute full lag/target features — skipping this run.")
        return

    # Cast category column to int64 to match Hopsworks bigint schema
    if "openweather_aqi_category" in features_df.columns:
        features_df["openweather_aqi_category"] = (
            features_df["openweather_aqi_category"].fillna(0).astype("int64")
        )

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs, df_for_schema=features_df)

    # Skip timestamps already stored, so a normal run only inserts the newest
    # hour(s) -- but a run after a missed hour or an outage naturally
    # backfills whatever's missing too, instead of just the latest point.
    try:
        existing_ts = set(
            fg.select(["timestamp"]).read()["timestamp"].astype(str)
        )
        features_df = features_df[~features_df["timestamp"].astype(str).isin(existing_ts)]
    except Exception as e:
        print(f"[feature_pipeline] Could not check existing timestamps ({e}); "
              f"inserting full window (Hopsworks will upsert on the timestamp "
              f"primary key, so duplicates are harmless, just slower).")

    if features_df.empty:
        print("[feature_pipeline] Nothing new to insert this run.")
        return

    fg.insert(features_df)
    print(f"[feature_pipeline] Inserted {len(features_df)} row(s) "
          f"({features_df['timestamp'].min()} to {features_df['timestamp'].max()}) "
          f"at {pd.Timestamp.utcnow()}")


def backfill_historical(start_date: str = None, chunk_days: int = 30):
    """
    Run ONCE, manually, before automation starts. Populates the feature store
    with historical data to train on.
    """
    config.validate_config()

    start_date = start_date or config.DATA_START_DATE
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp.utcnow().tz_localize(None)

    # --- 1. Pollution/AQI (OpenWeather) — chunked in 30-day windows ---
    raw_chunks = []
    chunk_start = start_ts
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), end_ts)
        chunk_df = fetch_historical_air_pollution(
            int(chunk_start.timestamp()), int(chunk_end.timestamp())
        )
        if not chunk_df.empty:
            raw_chunks.append(chunk_df)
            print(f"[backfill] Pollution: fetched {len(chunk_df)} rows for "
                  f"{chunk_start.date()} to {chunk_end.date()}")
        chunk_start = chunk_end

    if not raw_chunks:
        print("[backfill] No pollution data returned for the requested range.")
        return

    pollution_df = pd.concat(raw_chunks, ignore_index=True).drop_duplicates(subset="timestamp")

    # --- 2. Weather (Open-Meteo) ---
    weather_df = fetch_openmeteo_historical_weather(
        start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")
    )
    print(f"[backfill] Weather: fetched {len(weather_df)} rows for "
          f"{start_ts.date()} to {end_ts.date()}")

    # --- 3. Merge ---
    merged_df = merge_pollution_and_weather(pollution_df, weather_df)
    if merged_df.empty:
        print("[backfill] Merge produced 0 rows — pollution and weather "
              "timestamps didn't overlap. Check both sources are UTC.")
        return

    merged_df = validate_raw_data(merged_df)
    if merged_df.empty:
        print("[backfill] All rows failed validation — check the data sources.")
        return

    features_df = build_feature_set(merged_df)

    # Cast category column to int64 to match Hopsworks bigint schema
    if "openweather_aqi_category" in features_df.columns:
        features_df["openweather_aqi_category"] = (
            features_df["openweather_aqi_category"].fillna(0).astype("int64")
        )

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs, df_for_schema=features_df)
    fg.insert(features_df)

    print(f"[backfill] Done. Inserted {len(features_df)} historical rows "
          f"covering {start_ts.date()} to {end_ts.date()}.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        start = sys.argv[2] if len(sys.argv) > 2 else None
        backfill_historical(start_date=start)
    else:
        run_hourly_feature_pipeline()