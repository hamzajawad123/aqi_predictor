"""Score models on absolute future AQI so they can be compared."""
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
    """Turn a predicted change back into AQI. shrinkage=0 means no change."""
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
    """Pick a shrink factor on validation. Never use test for this."""
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
    """Guess that AQI later equals AQI now."""
    return evaluate(
        test_df[absolute_target_col],
        test_df["aqi"],
        "Persistence Baseline",
    )


def beats_persistence(model_metrics: dict, baseline_metrics: dict) -> bool:
    """Must beat persistence on RMSE, MAE and R²."""
    return (
        model_metrics["RMSE"] < baseline_metrics["RMSE"]
        and model_metrics["MAE"] < baseline_metrics["MAE"]
        and model_metrics["R2"] > baseline_metrics["R2"]
    )


def pick_best_candidate(results: list[dict], baseline: dict) -> dict | None:
    """Best model that beats persistence. None if nobody does."""
    eligible = [r for r in results if r["model"] != "Persistence Baseline"
                and beats_persistence(r, baseline)]
    if not eligible:
        return None
    eligible.sort(key=lambda r: (r["RMSE"], r["MAE"], -r["R2"]))
    return eligible[0]
