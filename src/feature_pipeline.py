"""Hourly feature pipeline. Fetch, merge, validate, write features to Hopsworks.

  python -m src.feature_pipeline
  python -m src.feature_pipeline backfill
  python -m src.feature_pipeline raw-snapshot
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

# Need 168h behind and 72h ahead, plus a little extra.
_MIN_LOOKBACK_HOURS = max(AQI_LAGS) + max(config.TARGET_HORIZONS) + 1
LOOKBACK_HOURS = _MIN_LOOKBACK_HOURS + 95  # 14 days

_INT64_COLS = (
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_smog_season",
    "openweather_aqi_category",
)


def _cast_for_hopsworks(features_df: pd.DataFrame) -> pd.DataFrame:
    """Make integer columns int64 for Hopsworks."""
    df = features_df.copy()
    for col in _INT64_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("int64")
    return df


def _unix_utc(ts: pd.Timestamp) -> int:
    """Unix seconds in UTC. Naive timestamps here are UTC, not local time."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp())


def _as_utc(series: pd.Series) -> pd.Series:
    """Timestamps as UTC so two equal hours are not treated as different."""
    return pd.to_datetime(series, utc=True)


def fetch_merged_lookback(lookback_hours: int = LOOKBACK_HOURS) -> pd.DataFrame:
    """Recent hours from both APIs, merged and checked."""
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
    # Drop future hours from Open-Meteo.
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
    """Full history from both APIs, merged and checked."""
    start_date = start_date or config.DATA_START_DATE
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
    """Save the raw merge to disk. No features, no Hopsworks."""
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
    """Hourly run: look back enough hours, then insert new rows."""
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

    # Insert new hours, and re-insert rows whose future AQI is now known.
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
    """One-time fill of history into Hopsworks."""
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
    """Build features from the local parquet and upload."""
    from src.utils.raw_io import load_raw_snapshot

    config.validate_config()
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
