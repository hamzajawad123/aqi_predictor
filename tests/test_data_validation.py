"""Data validation tests."""
import pandas as pd
import numpy as np
from src.utils.data_validation import validate_raw_data, DataValidationError


def _clean_df(n=20):
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": ts, "aqi": np.random.uniform(20, 200, n),
        "pm2_5": np.random.uniform(10, 150, n), "pm10": np.random.uniform(10, 200, n),
        "co": np.random.uniform(100, 500, n), "no2": np.random.uniform(5, 40, n),
        "o3": np.random.uniform(10, 60, n), "so2": np.random.uniform(1, 20, n),
        "nh3": np.random.uniform(1, 10, n), "temperature": np.random.uniform(5, 45, n),
        "humidity": np.random.uniform(20, 90, n), "wind_speed": np.random.uniform(0, 10, n),
        "wind_deg": np.random.uniform(0, 360, n), "pressure": np.random.uniform(990, 1020, n),
    })


def test_clean_data_passes_through_unchanged():
    df = _clean_df()
    result = validate_raw_data(df)
    assert len(result) == len(df)


def test_duplicate_timestamps_dropped():
    df = _clean_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = validate_raw_data(df)
    assert result["timestamp"].duplicated().sum() == 0
    assert len(result) == len(df) - 1


def test_out_of_range_values_dropped():
    df = _clean_df()
    df.loc[0, "pressure"] = -999  # impossible value
    result = validate_raw_data(df)
    assert len(result) == len(df) - 1
    assert -999 not in result["pressure"].values


def test_null_rows_dropped():
    df = _clean_df()
    df.loc[0, "humidity"] = None
    result = validate_raw_data(df)
    assert result["humidity"].isnull().sum() == 0
    assert len(result) == len(df) - 1


def test_missing_required_column_detected():
    df = _clean_df().drop(columns=["wind_speed"])
    result = validate_raw_data(df)  # should not crash, just warn
    assert "wind_speed" not in result.columns


def test_raise_on_error_mode():
    df = _clean_df()
    df.loc[0, "pressure"] = -999
    try:
        validate_raw_data(df, raise_on_error=True)
        assert False, "expected DataValidationError to be raised"
    except DataValidationError:
        pass
