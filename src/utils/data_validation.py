"""Check raw data before it goes into Hopsworks."""
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp", "aqi", "pm2_5", "pm10", "co", "no2", "o3", "so2", "nh3",
    "temperature", "humidity", "wind_speed", "wind_deg", "pressure",
]

# Wide ranges on purpose so real smog days are not dropped.
VALID_RANGES = {
    "aqi": (0, 1000),
    "pm2_5": (0, 2000),        # µg/m³
    "pm10": (0, 2000),
    "co": (0, 50000),
    "no2": (0, 2000),
    "o3": (0, 2000),
    "so2": (0, 2000),
    "nh3": (0, 2000),
    "temperature": (-30, 60),  # °C
    "humidity": (0, 100),      # %
    "wind_speed": (0, 150),    # km/h
    "wind_deg": (0, 360),
    "pressure": (850, 1100),   # hPa
}


class DataValidationError(Exception):
    """Bad data that should stop the insert."""


def validate_raw_data(df: pd.DataFrame, raise_on_error: bool = False) -> pd.DataFrame:
    """Drop bad rows. One bad hour should not kill the whole run."""
    issues = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        if raise_on_error:
            raise DataValidationError("; ".join(issues))
        return df

    n_dupes = df["timestamp"].duplicated().sum()
    if n_dupes > 0:
        issues.append(f"{n_dupes} duplicate timestamp(s) found — dropping duplicates")
        df = df.drop_duplicates(subset="timestamp", keep="first")

    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        issues.append(f"Null values found: {cols_with_nulls.to_dict()} — dropping affected rows")
        df = df.dropna(subset=REQUIRED_COLUMNS)

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
