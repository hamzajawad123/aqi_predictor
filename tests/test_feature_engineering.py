"""
Basic unit tests for feature engineering — run with: pytest tests/
Keeps the pipeline honest: if you change feature logic, these catch
accidental breakage before it reaches the feature store.
"""
import pandas as pd
from src.utils.feature_engineering import (
    add_time_features,
    add_lag_features,
    add_change_rate,
    build_feature_set,
)


def _sample_df(n=300):
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": ts,
        "aqi": [50 + (i % 24) for i in range(n)],
        "pm2_5": [20.0 + (i % 10) for i in range(n)],
        "wind_speed": [3.0 + (i % 5) for i in range(n)],
        "humidity": [40 + (i % 20) for i in range(n)],
    })


def test_add_time_features_creates_cyclical_columns():
    df = add_time_features(_sample_df())
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend"]:
        assert col in df.columns


def test_add_lag_features_shifts_correctly():
    df = add_lag_features(_sample_df(), col="aqi", lags=(1,))
    assert df["aqi_lag_1h"].iloc[1] == df["aqi"].iloc[0]


def test_add_change_rate_is_diff():
    df = add_change_rate(_sample_df())
    expected = df["aqi"].iloc[1] - df["aqi"].iloc[0]
    assert df["aqi_change_rate_1h"].iloc[1] == expected


def test_build_feature_set_drops_na_rows():
    df = build_feature_set(_sample_df())
    assert df.isna().sum().sum() == 0
    assert len(df) > 0
