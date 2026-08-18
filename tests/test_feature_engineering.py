"""
Basic unit tests for feature engineering — run with: pytest tests/
Keeps the pipeline honest: if you change feature logic, these catch
accidental breakage before it reaches the feature store.
"""
import pandas as pd
import numpy as np
from src.utils.feature_engineering import (
    add_time_features,
    add_lag_features,
    add_change_rate,
    build_feature_set,
    to_hourly_grid,
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
    for h in (24, 48, 72):
        assert f"aqi_target_{h}h" in df.columns
        assert f"aqi_delta_{h}h" in df.columns


def test_serving_mode_keeps_rows_whose_targets_are_unknown():
    """The current hour has no 72h future yet, and it's the row we predict from."""
    raw = _sample_df()
    serving = build_feature_set(raw, is_training=False)
    training = build_feature_set(raw, is_training=True)

    assert len(serving) > len(training)
    # the most recent usable hour survives, with real lags but no target
    latest = serving.iloc[-1]
    assert latest["timestamp"] == raw["timestamp"].iloc[-1]
    assert pd.notna(latest["aqi_lag_168h"])
    assert pd.isna(latest["aqi_target_72h"])


def test_serving_mode_returns_same_columns_as_training():
    """The feature group schema is built from this frame, so it must not vary."""
    raw = _sample_df()
    assert list(build_feature_set(raw, is_training=False).columns) == \
        list(build_feature_set(raw, is_training=True).columns)


def test_serving_mode_requires_complete_features():
    raw = _sample_df()
    serving = build_feature_set(raw, is_training=False)
    feature_cols = [c for c in serving.columns
                    if not c.startswith(("aqi_target_", "aqi_delta_"))]
    assert serving[feature_cols].isna().sum().sum() == 0


def test_hourly_lookback_window_can_yield_trainable_rows():
    """
    Regression: at LOOKBACK_HOURS=200 a row could not have both aqi_lag_168h
    (needs 168h behind) and aqi_target_72h (needs 72h ahead), so the hourly
    pipeline built an empty frame and inserted nothing, every run.
    """
    from src.feature_pipeline import LOOKBACK_HOURS

    window = _sample_df(n=LOOKBACK_HOURS)
    assert len(build_feature_set(window, is_training=True)) > 0


def test_to_hourly_grid_fills_missing_hours():
    df = _sample_df(n=50)
    gappy = df.drop(index=[10, 11, 12]).reset_index(drop=True)  # 3-hour outage

    grid = to_hourly_grid(gappy)
    steps = grid["timestamp"].diff().dt.total_seconds().div(3600).dropna()
    assert (steps == 1).all()
    assert len(grid) == 50
    # a short outage gets interpolated rather than dropped
    assert grid["aqi"].notna().all()


def test_to_hourly_grid_leaves_long_outages_missing():
    df = _sample_df(n=60)
    gappy = df.drop(index=range(10, 30)).reset_index(drop=True)  # 20-hour outage

    grid = to_hourly_grid(gappy, interpolate_limit=6)
    assert grid["aqi"].isna().sum() == 20  # not invented


def test_targets_span_real_hours_after_gaps():
    """A row-position shift on a gappy frame silently spans the wrong horizon."""
    df = _sample_df(n=200)
    gappy = df.drop(index=range(50, 60)).reset_index(drop=True)  # 10-hour outage

    built = build_feature_set(gappy)
    lookup = df.set_index("timestamp")["aqi"]
    for h in (24, 48, 72):
        sample = built.head(20)
        expected = [lookup.get(ts + pd.Timedelta(hours=h)) for ts in sample["timestamp"]]
        np.testing.assert_allclose(sample[f"aqi_target_{h}h"].to_numpy(float),
                                   np.array(expected, dtype=float))


def test_log_pollutants_applied():
    from src.utils.feature_engineering import add_log_pollutants
    raw = _sample_df()
    logged = add_log_pollutants(raw)
    # pm2_5=20 -> log1p(20)
    assert abs(logged["pm2_5"].iloc[0] - np.log1p(20.0)) < 1e-9
