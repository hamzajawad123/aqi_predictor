"""Evaluation helper tests."""
import numpy as np
import pandas as pd
import pytest

from src.utils.evaluation import (
    beats_persistence,
    evaluate,
    fit_delta_shrinkage,
    persistence_baseline,
    reconstruct_absolute,
)
from src.utils.sequences import make_sequences


def test_reconstruct_absolute_applies_shrinkage():
    aqi_now = np.array([100.0, 200.0])
    delta = np.array([20.0, -40.0])

    np.testing.assert_allclose(reconstruct_absolute(aqi_now, delta), [120.0, 160.0])
    np.testing.assert_allclose(reconstruct_absolute(aqi_now, delta, 0.5), [110.0, 180.0])
    # shrink=0 means no change
    np.testing.assert_allclose(reconstruct_absolute(aqi_now, delta, 0.0), aqi_now)


def test_persistence_baseline_equals_zero_delta():
    df = pd.DataFrame({"aqi": [100.0, 150.0, 200.0], "aqi_target_24h": [110.0, 140.0, 260.0]})
    baseline = persistence_baseline(df, "aqi_target_24h")
    zero_delta = evaluate(df["aqi_target_24h"], reconstruct_absolute(df["aqi"], np.zeros(3), 0.0), "x")
    assert baseline["RMSE"] == pytest.approx(zero_delta["RMSE"])


def test_shrinkage_collapses_useless_predictions_to_persistence():
    rng = np.random.default_rng(0)
    aqi_now = rng.uniform(50, 300, 500)
    true_delta = rng.normal(0, 20, 500)
    noise_delta = rng.normal(0, 60, 500)

    lam, diag = fit_delta_shrinkage(aqi_now, aqi_now + true_delta, noise_delta)
    assert lam < 0.3
    assert diag["val_rmse"] <= diag["val_persistence_rmse"]


def test_shrinkage_keeps_a_good_signal():
    rng = np.random.default_rng(1)
    aqi_now = rng.uniform(50, 300, 500)
    true_delta = rng.normal(0, 20, 500)
    good_delta = true_delta + rng.normal(0, 2, 500)

    lam, _ = fit_delta_shrinkage(aqi_now, aqi_now + true_delta, good_delta)
    assert lam > 0.8


def test_beats_persistence_requires_all_three_metrics():
    baseline = {"RMSE": 90.0, "MAE": 50.0, "R2": 0.5}
    assert beats_persistence({"RMSE": 80.0, "MAE": 45.0, "R2": 0.6}, baseline)
    # Must win on all three scores
    assert not beats_persistence({"RMSE": 80.0, "MAE": 55.0, "R2": 0.6}, baseline)


def test_make_sequences_window_ends_on_prediction_origin():
    n, seq_len = 30, 4
    df = pd.DataFrame({
        "aqi": np.arange(n, dtype=float),
        "pm2_5": np.arange(n, dtype=float) * 2,
        "aqi_delta_24h": np.arange(n, dtype=float) * -1,
        "aqi_target_24h": np.arange(n, dtype=float) + 5,
    })
    anchor = df["aqi"].to_numpy(float)
    abs_target = df["aqi_target_24h"].to_numpy(float)

    X, y, anchor_out, abs_out = make_sequences(
        df, ["aqi", "pm2_5"], "aqi_delta_24h", seq_len=seq_len,
        anchor=anchor, absolute_target=abs_target,
    )

    assert X.shape == (n - seq_len + 1, seq_len, 2)
    # Window ends on the hour we predict from
    for k in (0, 5, len(y) - 1):
        origin = k + seq_len - 1
        np.testing.assert_allclose(X[k][-1], [df["aqi"][origin], df["pm2_5"][origin]])
        assert y[k] == df["aqi_delta_24h"][origin]
        assert anchor_out[k] == anchor[origin]
        assert abs_out[k] == abs_target[origin]


def test_recurrent_anchor_is_not_scaled():
    """Anchor AQI must stay unscaled."""
    from sklearn.preprocessing import StandardScaler

    n, seq_len = 40, 4
    feature_cols = ["aqi", "pm2_5"]
    df = pd.DataFrame({
        "aqi": np.linspace(100, 300, n),
        "pm2_5": np.linspace(10, 90, n),
        "aqi_delta_24h": np.linspace(-20, 20, n),
    })
    anchor = df["aqi"].to_numpy(float)

    scaled = df.copy()
    scaled[feature_cols] = StandardScaler().fit_transform(scaled[feature_cols])

    _, _, anchor_out, _ = make_sequences(
        scaled, feature_cols, "aqi_delta_24h", seq_len=seq_len, anchor=anchor,
    )
    # Scaled AQI would sit near 0.
    assert anchor_out.min() >= 100.0
