"""
Training Pipeline
==================
Runs DAILY via .github/workflows/training_pipeline.yml

1. Fetch historical (features, targets) from the Hopsworks Feature Store.
2. Split chronologically, aligned to season boundaries (see chronological_split).
3. Train + evaluate ALL 7 models, deliberately ordered to match the brief's
   "from statistical modelling to deep learning" spectrum:
   Persistence baseline -> Prophet (classical statistical) -> Ridge ->
   Random Forest -> XGBoost -> LightGBM -> LSTM -> GRU (deep learning).
4. Use TimeSeriesSplit for all cross-validation (never shuffle time-series data),
   with Optuna driving the hyperparameter search for each tabular model.
5. Evaluate the winning TABULAR model two extra ways for the report:
   - stratified by season (smog vs. normal) — proves it works when it matters most
   - by forecast horizon (24h/48h/72h) — shows how accuracy degrades further out
6. Run SHAP on the best tree-based model for explainability.
7. Push the best-performing TABULAR 72h model to the Hopsworks Model Registry.

IMPORTANT — why Prophet/LSTM/GRU are compared but not registered/served:
Prophet takes a `ds` (date) column and forecasts by date; LSTM/GRU take a
windowed 3D input (samples, timesteps, features) built by their own
make_sequences(). The tabular models (Ridge/RandomForest/XGBoost/LightGBM)
take a flat 2D row instead — which is what api/main.py's single-row /predict
endpoint is built around. So: all 7 appear in the comparison table
(results_df), but only a tabular model is ever selected for the Model
Registry / FastAPI serving path. This is a deliberate, documented scope
boundary, not an oversight.
"""
import os
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor  # used only by multi_horizon_comparison's quick untuned check

from src import config
from src.utils.hopsworks_utils import get_feature_store, get_model_registry
from src.utils.optuna_tuning import tune_ridge, tune_random_forest, tune_xgboost, tune_lightgbm

TARGET_COL = "aqi_target_72h"  # primary target — the project's headline 3-day forecast
ALL_TARGET_COLS = [f"aqi_target_{h}h" for h in config.TARGET_HORIZONS]
DROP_COLS = ["timestamp"] + ALL_TARGET_COLS  # never feed target columns in as features
TABULAR_MODEL_NAMES = {"Ridge", "RandomForest", "XGBoost", "LightGBM"}
TREE_MODEL_NAMES = {"RandomForest", "XGBoost", "LightGBM"}  # TreeSHAP-compatible


def load_training_data() -> pd.DataFrame:
    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME,
                               version=config.FEATURE_GROUP_VERSION)
    return fg.read()


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _snap_to_june_first(ts: pd.Timestamp) -> pd.Timestamp:
    """Most recent 1-June on or before `ts`. Used so every partition boundary
    falls mid-year, never inside Lahore's Oct-Jan smog season."""
    year = ts.year if (ts.month, ts.day) >= (6, 1) else ts.year - 1
    return pd.Timestamp(year=year, month=6, day=1)


def chronological_split_by_fraction(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    """Simple percentage-based fallback split — NEVER shuffles time-series
    data, but doesn't guarantee full seasons per partition. Used automatically
    when there isn't enough history for the season-aligned split below."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def chronological_split(df: pd.DataFrame):
    """
    Season-aligned chronological split (the project's final decision):
    - Test  = most recent ~1 year, boundary snapped to 1 June
    - Val   = the 1 year before that
    - Train = everything older

    Snapping to 1 June (not 1 Jan) means every partition is a clean
    June-to-June block, so smog season (Oct-Jan) always sits safely in the
    middle of a partition instead of being split across two of them.

    Falls back to a plain 70/15/15 split if there isn't at least ~2 years of
    data yet (e.g. while testing the pipeline with a short backfill).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    max_date = df["timestamp"].max()

    test_start = _snap_to_june_first(max_date - pd.DateOffset(years=1))
    val_start = _snap_to_june_first(test_start - pd.DateOffset(years=1))

    train_df = df[df["timestamp"] < val_start]
    val_df = df[(df["timestamp"] >= val_start) & (df["timestamp"] < test_start)]
    test_df = df[df["timestamp"] >= test_start]

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("[training_pipeline] Not enough history yet for a season-aligned "
              "split — falling back to a simple 70/15/15 split.")
        return chronological_split_by_fraction(df)

    if len(train_df) < 180 * 24:  # less than ~6 months of hourly rows
        print(f"[training_pipeline] WARNING: train set is only {len(train_df)} rows "
              f"(~{len(train_df) // 24} days) — the season-aligned split reserves a "
              f"fixed 2 years for val+test, so with under ~4 years of total "
              f"history, train can end up small or miss a full smog season. "
              f"Backfill more history if possible (config.DATA_START_DATE).")

    print(f"[training_pipeline] Season-aligned split — "
          f"train: {df['timestamp'].min().date()} to {val_start.date()}, "
          f"val: {val_start.date()} to {test_start.date()}, "
          f"test: {test_start.date()} to {max_date.date()}")
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, name: str) -> dict:
    return {
        "model": name,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def persistence_baseline(test_df: pd.DataFrame, target_col: str = TARGET_COL) -> dict:
    """'AQI in N hours = AQI right now' — every real model must beat this."""
    y_true = test_df[target_col]
    y_pred = test_df["aqi"]
    return evaluate(y_true, y_pred, "Persistence Baseline")


def stratified_evaluation(model, test_df: pd.DataFrame, feature_cols: list,
                           target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Smog-season vs. normal-season metrics on the SAME trained model — proves
    (or disproves) that it holds up during Lahore's harshest AQI period,
    rather than hiding behind one blended number.
    """
    rows = []
    for label, subset in [("Normal season", test_df[test_df["is_smog_season"] == 0]),
                          ("Smog season", test_df[test_df["is_smog_season"] == 1]),
                          ("Overall", test_df)]:
        if len(subset) == 0:
            continue
        preds = model.predict(subset[feature_cols])
        result = evaluate(subset[target_col], preds, label)
        result["n_rows"] = len(subset)
        rows.append(result)
    return pd.DataFrame(rows)


def multi_horizon_comparison(train_df: pd.DataFrame, test_df: pd.DataFrame,
                              feature_cols: list) -> pd.DataFrame:
    """
    Quick (untuned) Random Forest per horizon, purely to show how accuracy
    degrades from 24h -> 48h -> 72h. This is intentionally lightweight
    (no Optuna tuning) — it's a supplementary comparison table for the
    report, not the model that gets registered (TARGET_COL / 72h uses the
    fully tuned model from train_and_evaluate() for that).
    """
    rows = []
    for h in config.TARGET_HORIZONS:
        target_col = f"aqi_target_{h}h"
        model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
        model.fit(train_df[feature_cols], train_df[target_col])
        preds = model.predict(test_df[feature_cols])
        result = evaluate(test_df[target_col], preds, f"RandomForest ({h}h ahead)")
        result["horizon_hours"] = h
        rows.append(result)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate():
    os.makedirs("reports", exist_ok=True)

    df = load_training_data()
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

    train_df, val_df, test_df = chronological_split(df)

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    tscv = TimeSeriesSplit(n_splits=5)
    results = [persistence_baseline(test_df)]
    fitted_models = {}

    # --- Prophet (classical statistical modelling — see train_prophet.py for
    # why this is placed first, right after the baseline: it's the actual
    # "statistical" end of the brief's requested statistical-to-deep-learning
    # spectrum, ahead of the regression-on-features models below). ---
    from src.train_prophet import train_prophet_model
    _, prophet_result = train_prophet_model(train_df, test_df)
    results.append(prophet_result)

    # --- Ridge ---
    ridge_model = tune_ridge(X_train, y_train, tscv)
    fitted_models["Ridge"] = ridge_model
    results.append(evaluate(y_test, ridge_model.predict(X_test), "Ridge"))

    # --- Random Forest ---
    rf_model = tune_random_forest(X_train, y_train, tscv)
    fitted_models["RandomForest"] = rf_model
    results.append(evaluate(y_test, rf_model.predict(X_test), "RandomForest"))

    # --- XGBoost ---
    xgb_model = tune_xgboost(X_train, y_train, tscv)
    fitted_models["XGBoost"] = xgb_model
    results.append(evaluate(y_test, xgb_model.predict(X_test), "XGBoost"))

    # --- LightGBM ---
    lgb_model = tune_lightgbm(X_train, y_train, tscv)
    fitted_models["LightGBM"] = lgb_model
    results.append(evaluate(y_test, lgb_model.predict(X_test), "LightGBM"))

    # --- LSTM & GRU (comparison-only — see module docstring for why these
    # aren't candidates for fitted_models / the Model Registry: they need a
    # windowed 3D input, incompatible with the flat-row serving path the
    # tabular models above share with api/main.py). ---
    from src.train_lstm import train_lstm_model
    from src.train_gru import train_gru_model

    _, lstm_result = train_lstm_model(train_df, val_df, test_df, feature_cols, TARGET_COL)
    results.append(lstm_result)

    _, gru_result = train_gru_model(train_df, val_df, test_df, feature_cols, TARGET_COL)
    results.append(gru_result)

    results_df = pd.DataFrame(results).sort_values("RMSE")
    print("\n=== Model comparison — ALL 7 MODELS (72h-ahead target) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("reports/model_comparison.csv", index=False)

    overall_best_name = results_df.iloc[0]["model"]

    # Registration/serving candidate is chosen from TABULAR models only —
    # see module docstring. If a sequence model actually scored lower RMSE
    # than every tabular model, say so plainly rather than silently ignoring it.
    tabular_results_df = results_df[results_df["model"].isin(TABULAR_MODEL_NAMES)]
    if tabular_results_df.empty:
        print("WARNING: no tabular model available to register — check the run.")
        return

    best_name = tabular_results_df.iloc[0]["model"]
    best_metrics = tabular_results_df.iloc[0].to_dict()
    if overall_best_name not in TABULAR_MODEL_NAMES and overall_best_name != "Persistence Baseline":
        print(f"\nNOTE: {overall_best_name} scored the lowest RMSE overall, but "
              f"isn't served (see module docstring) — registering the best "
              f"TABULAR model instead: {best_name}.")

    baseline_rmse = next(r["RMSE"] for r in results if r["model"] == "Persistence Baseline")
    if best_metrics["RMSE"] >= baseline_rmse:
        print(f"WARNING: best tabular model ({best_name}, RMSE={best_metrics['RMSE']:.2f}) "
              f"did not beat the persistence baseline (RMSE={baseline_rmse:.2f}) — "
              f"revisit features before registering.")
        return

    best_model = fitted_models[best_name]

    # --- Stratified (smog vs. normal season) evaluation ---
    strat_df = stratified_evaluation(best_model, test_df, feature_cols)
    print(f"\n=== Stratified evaluation ({best_name}) ===")
    print(strat_df.to_string(index=False))
    strat_df.to_csv("reports/model_comparison_stratified.csv", index=False)

    # --- Multi-horizon (24h/48h/72h) comparison ---
    horizon_df = multi_horizon_comparison(train_df, test_df, feature_cols)
    print("\n=== Multi-horizon comparison (quick Random Forest per horizon) ===")
    print(horizon_df.to_string(index=False))
    horizon_df.to_csv("reports/model_comparison_by_horizon.csv", index=False)

    # --- Explainability + registry ---
    explain_with_shap(best_model, X_test, best_name)
    register_model(best_model, best_name, best_metrics, feature_cols)


def explain_with_shap(model, X_test, model_name: str, max_samples: int = 2000):
    """
    TreeSHAP for tree-based models; save a summary plot for the report.
    Sampled to at most `max_samples` rows — standard SHAP practice, and
    necessary in practice: TreeExplainer over a full multi-year test set
    (tens of thousands of rows) is needlessly slow for a summary plot whose
    job is just to show overall feature-importance patterns, not explain
    every single row.
    """
    if model_name in TREE_MODEL_NAMES:
        X_sample = X_test.sample(n=min(max_samples, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False)
        import matplotlib.pyplot as plt
        plt.savefig("reports/shap_summary.png", bbox_inches="tight")
        plt.close()
        print(f"[training_pipeline] SHAP summary plot saved to reports/shap_summary.png "
              f"(sampled {len(X_sample)} of {len(X_test)} test rows)")
    else:
        print(f"[training_pipeline] SHAP skipped — {model_name} isn't tree-based "
              f"(TreeExplainer only). Use shap.LinearExplainer for Ridge if needed.")


def register_model(model, name: str, metrics: dict, feature_cols: list):
    import joblib

    os.makedirs("model_artifact", exist_ok=True)
    model_path = f"model_artifact/{name}.joblib"
    joblib.dump(model, model_path)

    mr = get_model_registry()
    hw_model = mr.python.create_model(
        name=config.MODEL_NAME,
        metrics={"RMSE": metrics["RMSE"], "MAE": metrics["MAE"], "R2": metrics["R2"]},
        description=f"AQI 72h forecaster — {name}",
    )
    hw_model.save("model_artifact")
    print(f"[training_pipeline] Registered {name} to Hopsworks Model Registry "
          f"(RMSE={metrics['RMSE']:.2f})")


if __name__ == "__main__":
    config.validate_config()
    train_and_evaluate()