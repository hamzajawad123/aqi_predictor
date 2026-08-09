"""
Data validation — runs BEFORE raw/merged data is written to the Hopsworks
Feature Store, in both the hourly pipeline and backfill.

Catches bad data early (the point of validation: a corrupt or out-of-range
reading should never silently become a training example) rather than letting
it flow into feature engineering and quietly degrade the model.
"""
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp", "aqi", "pm2_5", "pm10", "co", "no2", "o3", "so2", "nh3",
    "temperature", "humidity", "wind_speed", "wind_deg", "pressure",
]

# Sanity-check ranges. Wide on purpose — these catch clearly broken data
# (sensor errors, unit mistakes, API glitches), not flag legitimately bad
# air days (Lahore smog season can genuinely push PM2.5/AQI very high).
VALID_RANGES = {
    "aqi": (0, 1000),          # our computed continuous EPA AQI (see aqi_calculation.py)
    "pm2_5": (0, 2000),        # µg/m³
    "pm10": (0, 2000),         # µg/m³
    "co": (0, 50000),          # µg/m³
    "no2": (0, 2000),          # µg/m³
    "o3": (0, 2000),           # µg/m³
    "so2": (0, 2000),          # µg/m³
    "nh3": (0, 2000),          # µg/m³
    "temperature": (-30, 60),  # °C — generous bounds for Lahore's climate
    "humidity": (0, 100),      # %
    "wind_speed": (0, 150),    # km/h — generous, covers extreme storms
    "wind_deg": (0, 360),      # degrees
    "pressure": (850, 1100),   # hPa — generous sea-level-adjusted bounds
}


class DataValidationError(Exception):
    """Raised when incoming data fails validation badly enough to block insertion."""


def validate_raw_data(df: pd.DataFrame, raise_on_error: bool = False) -> pd.DataFrame:
    """
    Validates a merged (pollution + weather) dataframe before it's written to
    the feature store. Returns a cleaned dataframe with invalid rows dropped
    (not the whole batch discarded) — one bad hour shouldn't block an entire
    backfill or hourly run.

    Set raise_on_error=True to hard-fail instead (e.g. if you want the GitHub
    Action to alert you rather than silently drop rows).
    """
    issues = []

    # 1. Required columns present
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        if raise_on_error:
            raise DataValidationError("; ".join(issues))
        return df  # can't do row-level checks without the columns existing

    # 2. No duplicate timestamps (would corrupt the feature group's primary key)
    n_dupes = df["timestamp"].duplicated().sum()
    if n_dupes > 0:
        issues.append(f"{n_dupes} duplicate timestamp(s) found — dropping duplicates")
        df = df.drop_duplicates(subset="timestamp", keep="first")

    # 3. No nulls in required columns
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        issues.append(f"Null values found: {cols_with_nulls.to_dict()} — dropping affected rows")
        df = df.dropna(subset=REQUIRED_COLUMNS)

    # 4. Value range checks — drop rows with out-of-range readings (sensor
    # errors / API glitches), not the whole batch
    rows_before = len(df)
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            out_of_range = ~df[col].between(lo, hi)
            if out_of_range.any():
                issues.append(f"{out_of_range.sum()} row(s) with {col} outside "
                               f"[{lo}, {hi}] — dropping")
                df = df[~out_of_range]
    rows_dropped = rows_before - len(df)

    if issues:
        message = "[data_validation] " + " | ".join(issues)
        print(message)
        if raise_on_error:
            raise DataValidationError(message)

    if rows_dropped > 0:
        print(f"[data_validation] Dropped {rows_dropped} invalid row(s) out of "
              f"{rows_before} — {len(df)} remain.")

    return df.reset_index(drop=True)
