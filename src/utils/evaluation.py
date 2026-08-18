"""
Shared evaluation helpers for training / Colab / reports.

All model families are scored on absolute future AQI (RMSE, MAE, R2) so
delta-trained models, Prophet (absolute), and persistence are comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred, name: str) -> dict:
    """Return RMSE / MAE / R2 for one named prediction series."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "model": name,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def reconstruct_absolute(aqi_now, y_delta_pred, shrinkage: float = 1.0):
    """
    Convert delta predictions back to absolute AQI: aqi_now + shrinkage * delta.

    shrinkage=1.0 is the raw model, shrinkage=0.0 collapses to persistence.
    """
    return (
        np.asarray(aqi_now, dtype=float)
        + float(shrinkage) * np.asarray(y_delta_pred, dtype=float)
    )


SHRINKAGE_GRID = tuple(float(x) for x in np.round(np.linspace(0.0, 1.0, 21), 2))


def fit_delta_shrinkage(
    aqi_now,
    y_absolute_true,
    y_delta_pred,
    grid=SHRINKAGE_GRID,
) -> tuple[float, dict]:
    """
    Fit the delta shrinkage factor on VALIDATION data — never on test.

    Predicted deltas are noisy, and an over-confident delta hurts the many
    hours where AQI barely moves (which is where persistence is near-exact).
    Scaling the predicted delta down trades a little responsiveness on big
    swings for accuracy on typical hours. lambda=0 reproduces persistence
    exactly and lambda=1 is the untouched model, so the search can only pick
    something that is at least as good as persistence *on validation*.

    Scored on relative RMSE + relative MAE against the lambda=0 point because
    the registry gate demands beating persistence on RMSE, MAE and R2 at once
    (for a fixed y_true, R2 is monotone in RMSE, so it needs no separate term).

    Returns (lambda, diagnostics).
    """
    eps = 1e-12
    base = evaluate(y_absolute_true, reconstruct_absolute(aqi_now, y_delta_pred, 0.0), "lam=0")
    best_lam, best_score, best_metrics = 0.0, 1.0, base
    for lam in grid:
        m = evaluate(y_absolute_true, reconstruct_absolute(aqi_now, y_delta_pred, lam), f"lam={lam}")
        score = 0.5 * (
            m["RMSE"] / max(base["RMSE"], eps) + m["MAE"] / max(base["MAE"], eps)
        )
        if score < best_score - eps:
            best_lam, best_score, best_metrics = float(lam), score, m

    return best_lam, {
        "shrinkage": best_lam,
        "val_rmse": best_metrics["RMSE"],
        "val_mae": best_metrics["MAE"],
        "val_r2": best_metrics["R2"],
        "val_persistence_rmse": base["RMSE"],
        "val_persistence_mae": base["MAE"],
    }


def persistence_baseline(test_df: pd.DataFrame, absolute_target_col: str) -> dict:
    """
    'AQI in N hours = AQI right now'. Equivalent to predicting delta = 0.
    Computed once per horizon and reused for every model/ensemble comparison.
    """
    return evaluate(
        test_df[absolute_target_col],
        test_df["aqi"],
        "Persistence Baseline",
    )


def beats_persistence(model_metrics: dict, baseline_metrics: dict) -> bool:
    """True only if the model beats persistence on RMSE, MAE, AND R2."""
    return (
        model_metrics["RMSE"] < baseline_metrics["RMSE"]
        and model_metrics["MAE"] < baseline_metrics["MAE"]
        and model_metrics["R2"] > baseline_metrics["R2"]
    )


def pick_best_candidate(results: list[dict], baseline: dict) -> dict | None:
    """
    Among candidates that beat persistence on all three metrics, pick lowest
    RMSE (then MAE, then highest R2). Returns None if nobody qualifies.
    """
    eligible = [r for r in results if r["model"] != "Persistence Baseline"
                and beats_persistence(r, baseline)]
    if not eligible:
        return None
    eligible.sort(key=lambda r: (r["RMSE"], r["MAE"], -r["R2"]))
    return eligible[0]
