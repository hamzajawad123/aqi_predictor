"""SHAP plots for the saved models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config
from src.utils.feature_engineering import build_feature_set
from src.utils.raw_io import load_raw_snapshot

OUT = Path(__file__).resolve().parent / "_report_figures"
OUT.mkdir(exist_ok=True)
CACHE = ROOT / ".hw_cache" / "models"
MAX_SAMPLES = 400
RF_SAMPLES = 150
RNG = 42
TREE_TYPES = (
    "XGBRegressor",
    "LGBMRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
)


def chronological_split_by_fraction(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def _snap_to_june_first(ts: pd.Timestamp) -> pd.Timestamp:
    year = ts.year if (ts.month, ts.day) >= (6, 1) else ts.year - 1
    return pd.Timestamp(year=year, month=6, day=1)


def chronological_split(df: pd.DataFrame):
    """Same split as training."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    max_date = df["timestamp"].max()
    test_start = _snap_to_june_first(max_date - pd.DateOffset(years=1))
    val_start = _snap_to_june_first(test_start - pd.DateOffset(years=1))
    train_df = df[df["timestamp"] < val_start]
    val_df = df[(df["timestamp"] >= val_start) & (df["timestamp"] < test_start)]
    test_df = df[df["timestamp"] >= test_start]
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        return chronological_split_by_fraction(df)
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _unwrap(model):
    if isinstance(model, dict) and model.get("type") == "mean_ensemble":
        return None, model["members"]
    return model, None


def _is_tree(model) -> bool:
    return type(model).__name__ in TREE_TYPES


def _mean_abs(shap_values, columns) -> pd.Series:
    vals = np.asarray(shap_values)
    if vals.ndim == 3:
        vals = vals[0]
    return pd.Series(np.abs(vals).mean(axis=0), index=columns).sort_values(ascending=False)


def _plot_summary(shap_values, X, out_png: Path, title: str):
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    fig = plt.gcf()
    fig.suptitle(title, y=1.02, fontsize=11)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_bar(mean_abs: pd.Series, out_png: Path, title: str):
    top = mean_abs.head(15).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(top.index.astype(str), top.values, color="#1B365D")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    raw = load_raw_snapshot()
    feats = build_feature_set(raw, is_training=True)
    start = pd.Timestamp(config.TRAIN_START_DATE)
    feats = feats[feats["timestamp"] >= start].copy()
    train_df, val_df, test_df = chronological_split(feats)
    records = {
        "train_start_date": str(config.TRAIN_START_DATE),
        "n_post_break": int(len(feats)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "test_range": f"{test_df['timestamp'].min()} -> {test_df['timestamp'].max()}",
        "max_samples": MAX_SAMPLES,
        "random_state": RNG,
        "horizons": {},
    }

    for h in (24, 48, 72):
        path = CACHE / f"aqi_forecaster_{h}h" / "1" / f"aqi_forecaster_{h}h.joblib"
        payload = joblib.load(path)
        feature_cols = payload["feature_cols"]
        model, members = _unwrap(payload["model"])
        X_test = test_df[feature_cols]
        n_take = RF_SAMPLES if type(model).__name__ == "RandomForestRegressor" else MAX_SAMPLES
        X_sample = X_test.sample(n=min(n_take, len(X_test)), random_state=RNG)
        print(f"horizon {h}h model={payload.get('model_name')} shap_rows={len(X_sample)}")
        hrec = {
            "registry_name": f"aqi_forecaster_{h}h",
            "model_name": payload.get("model_name"),
            "target_type": payload.get("target_type"),
            "shrinkage": payload.get("shrinkage"),
            "n_shap_rows": int(len(X_sample)),
            "trees": [],
        }

        jobs = []
        if members is None:
            jobs.append((payload.get("model_name") or type(model).__name__, model, f"shap_{h}h"))
        else:
            for name, member in members.items():
                mobj = member["model"] if isinstance(member, dict) else member
                jobs.append((name, mobj, f"shap_{h}h_{name}"))

        for label, mobj, stem in jobs:
            if not _is_tree(mobj):
                hrec["trees"].append({"name": label, "skipped": type(mobj).__name__})
                continue
            print(f"  SHAP {label} ({type(mobj).__name__}) ...")
            if type(mobj).__name__ == "RandomForestRegressor":
                X_use = X_sample.sample(n=min(RF_SAMPLES, len(X_sample)), random_state=RNG)
            else:
                X_use = X_sample
            explainer = shap.TreeExplainer(mobj)
            sv = explainer.shap_values(X_use)
            mean_abs = _mean_abs(sv, feature_cols)
            _plot_summary(sv, X_use, OUT / f"{stem}_beeswarm.png", f"SHAP summary — {h}h {label}")
            _plot_bar(mean_abs, OUT / f"{stem}_bar.png", f"Mean |SHAP| — {h}h {label}")
            print(f"  saved {stem}")
            hrec["trees"].append(
                {
                    "name": label,
                    "class": type(mobj).__name__,
                    "n_shap_rows": int(len(X_use)),
                    "top10": [
                        {"feature": str(k), "mean_abs_shap": float(v)}
                        for k, v in mean_abs.head(10).items()
                    ],
                    "beeswarm": f"{stem}_beeswarm.png",
                    "bar": f"{stem}_bar.png",
                }
            )
        records["horizons"][str(h)] = hrec

    (OUT / "shap_stats.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("wrote", OUT / "shap_stats.json")


if __name__ == "__main__":
    main()
