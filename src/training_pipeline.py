"""
Training Pipeline
==================
Runs DAILY via .github/workflows/training_pipeline.yml

For each horizon in config.TARGET_HORIZONS (24h / 48h / 72h = day 1 / 2 / 3):
1. Train + Optuna-tune tabular models on aqi_delta_{h}h
2. Train Prophet on absolute aqi (level series), fitted through end of val
3. Train LSTM/GRU on delta; reconstruct absolute for metrics
4. Add a shrunk variant of every delta model, with the shrinkage factor fitted
   on VALIDATION (0 = persistence, 1 = raw model)
5. Build simple mean ensembles on reconstructed absolute predictions
6. Compare everyone to that horizon's persistence baseline (RMSE+MAE+R2)
7. Register ONLY the single best model that beats persistence on all 3
   metrics, as aqi_forecaster_{h}h

Set config.TRAIN_START_DATE to train on the recent regime only.
SHAP runs on the winning tree model per horizon when applicable.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import TimeSeriesSplit

from src import config
from src.utils.evaluation import (
    evaluate,
    fit_delta_shrinkage,
    persistence_baseline,
    reconstruct_absolute,
    pick_best_candidate,
)
from src.utils.hopsworks_utils import get_feature_store, get_model_registry
from src.utils.optuna_tuning import tune_ridge, tune_random_forest, tune_xgboost, tune_lightgbm
from src.train_prophet import train_prophet_model
from src.train_lstm import train_lstm_model
from src.train_gru import train_gru_model

TABULAR_MODEL_NAMES = {"Ridge", "RandomForest", "XGBoost", "LightGBM"}
TREE_MODEL_NAMES = {"RandomForest", "XGBoost", "LightGBM"}
# /predict feeds one flat feature row: Prophet needs a `ds` series and the
# recurrent nets need a 24-hour window, so neither can be served through it.
SERVEABLE_KINDS = {"tabular", "ensemble"}


def load_training_data() -> pd.DataFrame:
    """
    Read the feature group, minus rows whose targets aren't known yet.

    The hourly pipeline writes each hour as soon as its features are complete,
    which is ~72h before its targets can exist, so the newest rows in the store
    legitimately carry NULL targets. They're the rows inference reads; for
    training they'd be NaN labels, so drop them here at the boundary rather
    than letting each model discover them differently.
    """
    fs = get_feature_store()
    fg = fs.get_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
    )
    df = fg.read()

    target_cols = [
        c for h in config.TARGET_HORIZONS
        for c in (f"aqi_target_{h}h", f"aqi_delta_{h}h")
    ]
    present = [c for c in target_cols if c in df.columns]
    before = len(df)
    df = df.dropna(subset=present).reset_index(drop=True)
    if before != len(df):
        print(
            f"[training_pipeline] Dropped {before - len(df)} row(s) whose "
            f"targets are still in the future ({len(df)} trainable rows)."
        )
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Drop timestamp + all absolute/delta target columns from model inputs."""
    drop = {"timestamp"}
    for h in config.TARGET_HORIZONS:
        drop.add(f"aqi_target_{h}h")
        drop.add(f"aqi_delta_{h}h")
    # Raw hour/month are redundant with cyclical encodings (kept historically)
    drop.update({"hour", "month", "openweather_aqi_category"})
    return [c for c in df.columns if c not in drop]


def _snap_to_june_first(ts: pd.Timestamp) -> pd.Timestamp:
    year = ts.year if (ts.month, ts.day) >= (6, 1) else ts.year - 1
    return pd.Timestamp(year=year, month=6, day=1)


def chronological_split_by_fraction(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def chronological_split(df: pd.DataFrame):
    """Season-aligned June→June chronological split (smog season stays intact)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    max_date = df["timestamp"].max()

    test_start = _snap_to_june_first(max_date - pd.DateOffset(years=1))
    val_start = _snap_to_june_first(test_start - pd.DateOffset(years=1))

    train_df = df[df["timestamp"] < val_start]
    val_df = df[(df["timestamp"] >= val_start) & (df["timestamp"] < test_start)]
    test_df = df[df["timestamp"] >= test_start]

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("[training_pipeline] Not enough history for season-aligned split "
              "- falling back to 70/15/15.")
        return chronological_split_by_fraction(df)

    print(
        f"[training_pipeline] Season-aligned split - "
        f"train: {df['timestamp'].min().date()} to {val_start.date()}, "
        f"val: {val_start.date()} to {test_start.date()}, "
        f"test: {test_start.date()} to {max_date.date()}"
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


SHRUNK_SUFFIX = "_shrunk"


def _ensemble_mean(pred_arrays: list[np.ndarray], name: str, y_true) -> tuple[np.ndarray, dict]:
    stacked = np.vstack(pred_arrays)
    mean_pred = stacked.mean(axis=0)
    return mean_pred, evaluate(y_true, mean_pred, name)


def _add_delta_candidate(
    candidates: dict,
    results: list[dict],
    *,
    name: str,
    kind: str,
    model,
    horizon: int,
    val_delta_pred,
    val_anchor,
    val_absolute_true,
    test_delta_pred,
    test_anchor,
    test_absolute_true,
) -> None:
    """
    Score one delta model on reconstructed absolute AQI and, when enabled, add
    a second candidate whose delta is shrunk by a factor fitted on validation.

    Both variants share the same fitted model — only the multiplier applied to
    its predicted delta differs, so the shrunk variant costs no extra training.
    """
    raw_pred = reconstruct_absolute(test_anchor, test_delta_pred)
    metrics = evaluate(test_absolute_true, raw_pred, name)
    metrics["horizon_hours"] = horizon
    metrics["shrinkage"] = 1.0
    results.append(metrics)
    candidates[name] = {
        "model": model,
        "kind": kind,
        "shrinkage": 1.0,
        "abs_pred": raw_pred,
        "metrics": metrics,
    }

    if not config.USE_DELTA_SHRINKAGE or val_delta_pred is None:
        return

    lam, diag = fit_delta_shrinkage(val_anchor, val_absolute_true, val_delta_pred)
    print(
        f"[training_pipeline] {name} {horizon}h shrinkage fitted on val: "
        f"lambda={lam:.2f} (val RMSE {diag['val_rmse']:.2f} vs persistence "
        f"{diag['val_persistence_rmse']:.2f})"
    )
    # lambda == 1 duplicates the raw row; lambda == 0 is exactly persistence,
    # which can never pass a strictly-better gate.
    if lam <= 0.0 or lam >= 1.0:
        return

    shrunk_name = f"{name}{SHRUNK_SUFFIX}"
    shrunk_pred = reconstruct_absolute(test_anchor, test_delta_pred, lam)
    shrunk_metrics = evaluate(test_absolute_true, shrunk_pred, shrunk_name)
    shrunk_metrics["horizon_hours"] = horizon
    shrunk_metrics["shrinkage"] = lam
    results.append(shrunk_metrics)
    candidates[shrunk_name] = {
        "model": model,
        "kind": kind,
        "shrinkage": lam,
        "abs_pred": shrunk_pred,
        "metrics": shrunk_metrics,
    }


def _build_artifact(winner: dict, candidates: dict, results: list[dict],
                    baseline: dict, feature_cols: list[str], horizon: int) -> dict | None:
    """
    Turn the winning comparison row into a registry artifact.

    If the overall winner can't be served through /predict (Prophet, LSTM, GRU)
    we fall back to the best serveable candidate that still beats persistence
    on all three metrics, and log the substitution.
    """
    name = winner["model"]
    cand = candidates.get(name)

    if cand is None or cand["kind"] not in SERVEABLE_KINDS:
        serveable_rows = [
            r for r in results
            if candidates.get(r["model"], {}).get("kind") in SERVEABLE_KINDS
        ]
        fallback = pick_best_candidate(serveable_rows, baseline)
        if fallback is None:
            print(
                f"[training_pipeline] Winner {name} is not serveable through "
                f"/predict and no serveable model beat persistence — skip registry."
            )
            return None
        print(
            f"[training_pipeline] Winner {name} is not serveable through "
            f"/predict; registering {fallback['model']} instead."
        )
        winner, name = fallback, fallback["model"]
        cand = candidates[name]

    return {
        "name": name,
        "model": cand["model"],
        "metrics": winner,
        "feature_cols": feature_cols,
        "horizon": horizon,
        "target_type": "delta",
        "shrinkage": cand["shrinkage"],
    }


def train_one_horizon(
    horizon: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Train all models for one horizon; return comparison + optional winner artifact."""
    delta_col = f"aqi_delta_{horizon}h"
    abs_col = f"aqi_target_{horizon}h"

    baseline = persistence_baseline(test_df, abs_col)
    baseline["horizon_hours"] = horizon
    baseline["shrinkage"] = 0.0  # persistence IS the delta=0 prediction
    results = [baseline]
    candidates: dict[str, dict] = {}

    X_train, y_train_delta = train_df[feature_cols], train_df[delta_col]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    y_test_abs = test_df[abs_col].to_numpy(dtype=float)
    aqi_test = test_df["aqi"].to_numpy(dtype=float)
    aqi_val = val_df["aqi"].to_numpy(dtype=float)
    y_val_abs = val_df[abs_col].to_numpy(dtype=float)

    tscv = TimeSeriesSplit(n_splits=5)

    # --- Prophet (absolute levels, fitted through end of val so it only ever
    # forecasts `horizon` hours past its own data) ---
    prophet_fit_df = pd.concat([train_df, val_df], ignore_index=True)
    _, prophet_result = train_prophet_model(
        prophet_fit_df, test_df, target_col=abs_col, horizon_hours=horizon
    )
    prophet_result["horizon_hours"] = horizon
    results.append(prophet_result)

    # --- Tabular on deltas ---
    for name, tuner in [
        ("Ridge", tune_ridge),
        ("RandomForest", tune_random_forest),
        ("XGBoost", tune_xgboost),
        ("LightGBM", tune_lightgbm),
    ]:
        model = tuner(X_train, y_train_delta, tscv)
        _add_delta_candidate(
            candidates, results,
            name=name, kind="tabular", model=model, horizon=horizon,
            val_delta_pred=model.predict(X_val),
            val_anchor=aqi_val,
            val_absolute_true=y_val_abs,
            test_delta_pred=model.predict(X_test),
            test_anchor=aqi_test,
            test_absolute_true=y_test_abs,
        )

    # --- LSTM / GRU on deltas (windowed, so their preds are seq_len-1 rows
    # shorter than the tabular ones and stay out of the ensembles) ---
    for name, trainer in [("LSTM", train_lstm_model), ("GRU", train_gru_model)]:
        model, _, meta = trainer(
            train_df, val_df, test_df, feature_cols, delta_col,
            absolute_target_col=abs_col, report_name=name,
        )
        _add_delta_candidate(
            candidates, results,
            name=name, kind="recurrent", model=(model, meta), horizon=horizon,
            val_delta_pred=meta["val_delta_pred"],
            val_anchor=meta["val_anchor"],
            val_absolute_true=meta["val_absolute_true"],
            test_delta_pred=meta["test_delta_pred"],
            test_anchor=meta["test_anchor"],
            test_absolute_true=meta["test_absolute_true"],
        )

    # --- Ensembles over the tabular candidates (raw and shrunk both eligible) ---
    tabular_ranked = [
        name for name, c in sorted(
            candidates.items(), key=lambda kv: kv[1]["metrics"]["RMSE"]
        )
        if c["kind"] == "tabular"
    ]
    # One variant per base model, so an ensemble can't be three copies of XGBoost
    seen_bases, members_top = set(), []
    for name in tabular_ranked:
        base = name.removesuffix(SHRUNK_SUFFIX)
        if base in seen_bases:
            continue
        seen_bases.add(base)
        members_top.append(name)
        if len(members_top) == 3:
            break

    for ens_name, members in [
        (f"Ensemble_top{len(members_top)}_tabular", members_top),
        ("Ensemble_XGB_LGBM", [n for n in tabular_ranked
                               if n.removesuffix(SHRUNK_SUFFIX) in {"XGBoost", "LightGBM"}][:2]),
    ]:
        if len(members) < 2 or ens_name in candidates:
            continue
        ens_pred, ens_metrics = _ensemble_mean(
            [candidates[m]["abs_pred"] for m in members], ens_name, y_test_abs
        )
        ens_metrics["horizon_hours"] = horizon
        ens_metrics["shrinkage"] = float(
            np.mean([candidates[m]["shrinkage"] for m in members])
        )
        results.append(ens_metrics)
        candidates[ens_name] = {
            "model": {
                "type": "mean_ensemble",
                "members": {
                    m: {"model": candidates[m]["model"],
                        "shrinkage": candidates[m]["shrinkage"]}
                    for m in members
                },
            },
            "kind": "ensemble",
            "shrinkage": 1.0,  # each member already carries its own factor
            "abs_pred": ens_pred,
            "metrics": ens_metrics,
        }

    results_df = pd.DataFrame(results).sort_values("RMSE")
    print(f"\n=== Horizon {horizon}h (day {horizon // 24}) — all models ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(f"reports/model_comparison_{horizon}h.csv", index=False)

    winner = pick_best_candidate(results, baseline)
    artifact = None
    if winner is None:
        print(
            f"[training_pipeline] No model beat persistence on all 3 metrics "
            f"for {horizon}h — nothing registered."
        )
    else:
        print(
            f"[training_pipeline] Winner for {horizon}h: {winner['model']} "
            f"(RMSE={winner['RMSE']:.2f}, MAE={winner['MAE']:.2f}, R2={winner['R2']:.3f})"
        )
        artifact = _build_artifact(winner, candidates, results, baseline,
                                  feature_cols, horizon)

        if artifact and artifact["name"].removesuffix(SHRUNK_SUFFIX) in TREE_MODEL_NAMES:
            explain_with_shap(
                artifact["model"],
                X_test,
                artifact["name"].removesuffix(SHRUNK_SUFFIX),
                out_path=f"reports/shap_summary_{horizon}h.png",
            )

    return {
        "horizon": horizon,
        "results_df": results_df,
        "baseline": baseline,
        "winner": winner,
        "artifact": artifact,
    }


def explain_with_shap(model, X_test, model_name: str, max_samples: int = 2000,
                      out_path: str = "reports/shap_summary.png"):
    if model_name not in TREE_MODEL_NAMES:
        return
    X_sample = X_test.sample(n=min(max_samples, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, show=False)
    import matplotlib.pyplot as plt
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[training_pipeline] SHAP saved to {out_path}")


def register_model(artifact: dict):
    """Register one horizon winner to Hopsworks Model Registry."""
    horizon = artifact["horizon"]
    name = artifact["name"]
    metrics = artifact["metrics"]
    model_registry_name = config.model_name_for_horizon(horizon)

    os.makedirs("model_artifact", exist_ok=True)
    shrinkage = float(artifact.get("shrinkage", 1.0))
    payload = {
        "model": artifact["model"],
        "feature_cols": artifact["feature_cols"],
        "horizon_hours": horizon,
        "target_type": artifact.get("target_type", "delta"),
        "model_name": name,
        # Serving must apply the same factor training scored with, or the
        # served prediction won't match the registered metrics.
        "shrinkage": shrinkage,
    }
    model_path = f"model_artifact/{model_registry_name}.joblib"
    joblib.dump(payload, model_path)

    mr = get_model_registry()
    hw_model = mr.python.create_model(
        name=model_registry_name,
        metrics={
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
        },
        description=(
            f"AQI {horizon}h (day {horizon // 24}) forecaster — {name} "
            f"(delta target, serve as aqi + {shrinkage:.2f} * predicted_delta)"
        ),
    )
    hw_model.save("model_artifact")
    print(
        f"[training_pipeline] Registered {name} as {model_registry_name} "
        f"(RMSE={metrics['RMSE']:.2f})"
    )


def train_and_evaluate():
    os.makedirs("reports", exist_ok=True)

    df = load_training_data()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    if config.TRAIN_START_DATE:
        cutoff = pd.Timestamp(config.TRAIN_START_DATE)
        before = len(df)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
        print(
            f"[training_pipeline] TRAIN_START_DATE={cutoff.date()} — using the "
            f"recent regime only: {len(df)} of {before} rows."
        )

    train_df, val_df, test_df = chronological_split(df)
    feature_cols = feature_columns(df)

    all_results = []
    for h in config.TARGET_HORIZONS:
        out = train_one_horizon(h, train_df, val_df, test_df, feature_cols)
        out["results_df"]["horizon_hours"] = h
        all_results.append(out["results_df"])
        if out["artifact"] is not None:
            register_model(out["artifact"])

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("reports/model_comparison.csv", index=False)
    combined.to_csv("reports/model_comparison_by_horizon.csv", index=False)
    print("\n=== Combined comparison (all horizons) ===")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    config.validate_config()
    train_and_evaluate()
