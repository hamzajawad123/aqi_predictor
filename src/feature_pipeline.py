"""
Feature Pipeline
================
Runs HOURLY via .github/workflows/feature_pipeline.yml

1. Fetch raw pollutant data (OpenWeather) + weather data (Open-Meteo).
2. Merge them on their shared UTC timestamp and validate.
3. Persist the raw merged snapshot to disk (for EDA / reproducibility).
4. Compute features (time-based, lags, rolling stats, AQI change rate, season flag).
5. Write the resulting row(s) to the Hopsworks Feature Store.

Also usable directly for:
  - BACKFILL:   python -m src.feature_pipeline backfill [YYYY-MM-DD]
  - RAW ONLY:   python -m src.feature_pipeline raw-snapshot [YYYY-MM-DD]
                (fetch + validate + save parquet; no FE, no Hopsworks)

IMPORTANT: pollution always comes from OpenWeather and weather always comes
from Open-Meteo, in BOTH this hourly path and the backfill path. Using
different sources for the same variable between backfill and live collection
is exactly the train/serving skew a feature store exists to prevent — so
don't swap either source in just one of the two functions below.
"""
from __future__ import annotations

import sys

import pandas as pd

from src import config
from src.utils.data_fetch import (
    fetch_historical_air_pollution,
    fetch_openmeteo_recent_weather,
    fetch_openmeteo_historical_weather,
    merge_pollution_and_weather,
)
from src.utils.feature_engineering import AQI_LAGS, build_feature_set
from src.utils.hopsworks_utils import get_feature_store, get_or_create_feature_group
from src.utils.data_validation import validate_raw_data
from src.utils.raw_io import save_raw_snapshot, upsert_raw_snapshot

# A row needs max(AQI_LAGS) hours BEHIND it for its lags and max(TARGET_HORIZONS)
# hours AHEAD of it for its targets, so a window has to span both before it can
# yield a single row that has each. At the previous 200h it could not: rows old
# enough for aqi_lag_168h were newer than 72h, so build_feature_set returned
# zero rows and every hourly run inserted nothing (verified: 200 rows in, 0 out).
# The margin on top absorbs late/missed upstream hours.
_MIN_LOOKBACK_HOURS = max(AQI_LAGS) + max(config.TARGET_HORIZONS) + 1
LOOKBACK_HOURS = _MIN_LOOKBACK_HOURS + 95  # 336h = 14 days

_INT64_COLS = (
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_smog_season",
    "openweather_aqi_category",
)


def _cast_for_hopsworks(features_df: pd.DataFrame) -> pd.DataFrame:
    """Cast integer-like columns to int64 so Hopsworks bigint schema matches."""
    df = features_df.copy()
    for col in _INT64_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("int64")
    return df


def _unix_utc(ts: pd.Timestamp) -> int:
    """
    UNIX seconds for OpenWeather. `.timestamp()` on a naive Timestamp is
    treated as *local* time, which shifts the requested window by the host
    UTC offset. Naive values here are UTC wall-clock hours, so localize
    them before converting.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp())


def _as_utc(series: pd.Series) -> pd.Series:
    """
    Normalize timestamps to UTC-aware so string/tz format cannot make two
    equal hours look different. Hopsworks may return
    '2026-08-18 05:00:00+00:00'; local frames are naive UTC. Comparing either
    as str would treat those as distinct and re-insert every stored hour.
    """
    return pd.to_datetime(series, utc=True)


def fetch_merged_lookback(lookback_hours: int = LOOKBACK_HOURS) -> pd.DataFrame:
    """
    Fetch a rolling lookback window from both APIs, merge on timestamp,
    clip to <= now, and validate. Returns an empty DataFrame on soft failure.
    """
    # Both forms are needed. .timestamp() on a tz-NAIVE Timestamp reads it as
    # local time, which would shift the requested window by the machine's UTC
    # offset, so the API bounds come from the tz-aware value. The fetchers
    # return tz-naive UTC though, so the frame has to be compared against the
    # naive one (mixing them raises "Cannot compare tz-naive and tz-aware").
    now_utc = pd.Timestamp.now(tz="UTC").floor("h")
    now = now_utc.tz_localize(None)
    lookback_start = now_utc - pd.Timedelta(hours=lookback_hours)

    pollution_df = fetch_historical_air_pollution(
        _unix_utc(lookback_start), _unix_utc(now_utc)
    )
    weather_df = fetch_openmeteo_recent_weather(
        past_days=(lookback_hours // 24) + 2
    )

    print(
        f"[feature_pipeline] Debug — pollution_df rows: {len(pollution_df)}, "
        f"weather_df rows: {len(weather_df)}"
    )

    if pollution_df.empty or weather_df.empty:
        print(
            "[feature_pipeline] One of the two sources returned no data this "
            "run (see debug output above) — skipping."
        )
        return pd.DataFrame()

    merged_df = merge_pollution_and_weather(pollution_df, weather_df)
    # Keep only the lookback window -- Open-Meteo's forecast_days=1 can hand
    # back a couple of hours past `now`; those aren't real observations yet.
    merged_df = merged_df[merged_df["timestamp"] <= now]

    if merged_df.empty:
        print(
            "[feature_pipeline] No overlapping timestamps between the two "
            "sources this run — skipping (will retry next hour)."
        )
        return pd.DataFrame()

    merged_df = validate_raw_data(merged_df)
    if merged_df.empty:
        print(
            "[feature_pipeline] All rows failed validation this run — skipping "
            "(will retry next hour)."
        )
        return pd.DataFrame()

    return merged_df


def fetch_merged_historical(
    start_date: str | None = None, chunk_days: int = 30
) -> pd.DataFrame:
    """
    Fetch full historical pollution (chunked) + weather, merge, and validate.
    Stops before feature engineering — callers decide whether to save raw,
    engineer features, and/or push to Hopsworks.
    """
    start_date = start_date or config.DATA_START_DATE
    # Same UTC rule as fetch_merged_lookback: naive .timestamp() is local time.
    start_utc = pd.Timestamp(start_date, tz="UTC")
    end_utc = pd.Timestamp.now(tz="UTC").floor("h")
    start_ts = start_utc.tz_localize(None)
    end_ts = end_utc.tz_localize(None)

    raw_chunks = []
    chunk_start = start_utc
    while chunk_start < end_utc:
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), end_utc)
        chunk_df = fetch_historical_air_pollution(
            _unix_utc(chunk_start), _unix_utc(chunk_end)
        )
        if not chunk_df.empty:
            raw_chunks.append(chunk_df)
            print(
                f"[backfill] Pollution: fetched {len(chunk_df)} rows for "
                f"{chunk_start.date()} to {chunk_end.date()}"
            )
        chunk_start = chunk_end

    if not raw_chunks:
        print("[backfill] No pollution data returned for the requested range.")
        return pd.DataFrame()

    pollution_df = pd.concat(raw_chunks, ignore_index=True).drop_duplicates(
        subset="timestamp"
    )

    weather_df = fetch_openmeteo_historical_weather(
        start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")
    )
    print(
        f"[backfill] Weather: fetched {len(weather_df)} rows for "
        f"{start_ts.date()} to {end_ts.date()}"
    )

    merged_df = merge_pollution_and_weather(pollution_df, weather_df)
    if merged_df.empty:
        print(
            "[backfill] Merge produced 0 rows — pollution and weather "
            "timestamps didn't overlap. Check both sources are UTC."
        )
        return pd.DataFrame()

    merged_df = validate_raw_data(merged_df)
    if merged_df.empty:
        print("[backfill] All rows failed validation — check the data sources.")
        return pd.DataFrame()

    return merged_df


def run_raw_snapshot(start_date: str | None = None, chunk_days: int = 30) -> None:
    """
    Step 1 entrypoint: fetch + validate historical merge, save to
    config.RAW_DATA_PATH, and exit. No feature engineering, no Hopsworks.
    """
    # Only OpenWeather is required for a raw fetch; Hopsworks keys are not.
    if not config.OPENWEATHER_API_KEY:
        raise EnvironmentError(
            "Missing OPENWEATHER_API_KEY. Copy .env.example to .env and fill in."
        )

    merged_df = fetch_merged_historical(start_date=start_date, chunk_days=chunk_days)
    if merged_df.empty:
        print("[raw-snapshot] Nothing to save.")
        return

    path = save_raw_snapshot(merged_df)
    null_counts = merged_df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    print(f"[raw-snapshot] Path: {path}")
    print(f"[raw-snapshot] Rows: {len(merged_df)}")
    print(
        f"[raw-snapshot] Range: {merged_df['timestamp'].min()} -> "
        f"{merged_df['timestamp'].max()}"
    )
    print(
        f"[raw-snapshot] Null columns:\n"
        f"{null_counts if len(null_counts) else '(none)'}"
    )


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

    The window is built with is_training=False, so the newest hours are kept
    even though their targets are still in the future — those are the rows the
    forecast is made from. Their targets land as NULL and are filled in by a
    later run, once the hours they refer to have actually happened.
    """
    config.validate_config()

    merged_df = fetch_merged_lookback(LOOKBACK_HOURS)
    if merged_df.empty:
        return

    upsert_raw_snapshot(merged_df)

    features_df = build_feature_set(merged_df, is_training=False)
    if features_df.empty:
        print(
            "[feature_pipeline] Not enough history in this window yet to "
            "compute full lag/target features — skipping this run."
        )
        return

    features_df = _cast_for_hopsworks(features_df)

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs, df_for_schema=features_df)

    # Skip timestamps already stored, so a normal run only inserts the newest
    # hour(s) -- but a run after a missed hour or an outage naturally
    # backfills whatever's missing too, instead of just the latest point.
    #
    # EXCEPT rows whose targets are now known. Every row is first written while
    # its future hasn't happened, so it lands with NULL targets; skipping it
    # forever after would leave it permanently untrainable and quietly stop the
    # training set from growing. Re-inserting it upserts on the timestamp
    # primary key, replacing the NULLs with the real future.
    target_cols = [f"aqi_target_{h}h" for h in config.TARGET_HORIZONS]
    has_targets = features_df[target_cols].notna().all(axis=1)
    try:
        existing_ts = set(
            _as_utc(fg.select(["timestamp"]).read()["timestamp"])
        )
        is_new = ~_as_utc(features_df["timestamp"]).isin(existing_ts)
        features_df = features_df[is_new | has_targets]
    except Exception as e:
        print(
            f"[feature_pipeline] Could not check existing timestamps ({e}); "
            f"inserting full window (Hopsworks will upsert on the timestamp "
            f"primary key, so duplicates are harmless, just slower)."
        )

    if features_df.empty:
        print("[feature_pipeline] Nothing new to insert this run.")
        return

    fg.insert(features_df)
    print(
        f"[feature_pipeline] Inserted {len(features_df)} row(s) "
        f"({features_df['timestamp'].min()} to {features_df['timestamp'].max()}) "
        f"at {pd.Timestamp.utcnow()}"
    )


def backfill_historical(start_date: str | None = None, chunk_days: int = 30):
    """
    Run ONCE, manually, before automation starts. Populates the feature store
    with historical data to train on. Also upserts the local raw snapshot.
    """
    config.validate_config()

    start_date = start_date or config.DATA_START_DATE
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp.utcnow().tz_localize(None)

    merged_df = fetch_merged_historical(start_date=start_date, chunk_days=chunk_days)
    if merged_df.empty:
        return

    upsert_raw_snapshot(merged_df)

    features_df = build_feature_set(merged_df)
    features_df = _cast_for_hopsworks(features_df)

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs, df_for_schema=features_df)
    fg.insert(features_df)

    print(
        f"[backfill] Done. Inserted {len(features_df)} historical rows "
        f"covering {start_ts.date()} to {end_ts.date()}."
    )


def push_features_from_raw():
    """
    Build the post-EDA feature set from the local raw snapshot and insert into
    Hopsworks (FEATURE_GROUP_VERSION, default v4). Skips re-fetching APIs.
    """
    from src.utils.raw_io import load_raw_snapshot

    config.validate_config()
    # Announced before the upload, not after: an unsaved .env edit once sent a
    # full rebuild into the previous feature group version, and the only
    # mention of the target came in the closing line.
    print(
        f"[push-features] Target: {config.FEATURE_GROUP_NAME} "
        f"v{config.FEATURE_GROUP_VERSION} (from FEATURE_GROUP_VERSION)"
    )
    merged_df = load_raw_snapshot()
    print(f"[push-features] Loaded raw snapshot: {len(merged_df)} rows")

    features_df = build_feature_set(merged_df)
    features_df = _cast_for_hopsworks(features_df)

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs, df_for_schema=features_df)
    fg.insert(features_df)
    print(
        f"[push-features] Inserted {len(features_df)} rows into "
        f"{config.FEATURE_GROUP_NAME} v{config.FEATURE_GROUP_VERSION}"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "raw-snapshot":
        start = sys.argv[2] if len(sys.argv) > 2 else None
        run_raw_snapshot(start_date=start)
    elif len(sys.argv) > 1 and sys.argv[1] == "backfill":
        start = sys.argv[2] if len(sys.argv) > 2 else None
        backfill_historical(start_date=start)
    elif len(sys.argv) > 1 and sys.argv[1] == "push-features":
        push_features_from_raw()
    else:
        run_hourly_feature_pipeline()
