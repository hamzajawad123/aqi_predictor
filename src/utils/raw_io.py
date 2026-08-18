"""
Local persistence for raw (pre-feature-engineering) pollution+weather data.

The hourly/backfill pipelines fetch and validate a merged frame, then call
these helpers so EDA and later FE always have a reproducible raw snapshot on
disk — not only the engineered Hopsworks table.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config

# Project root (…/aqi-predictor), not the caller's CWD. Notebooks run from
# notebooks/ so a relative "data/raw/…" path would otherwise miss the file.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path | None = None) -> Path:
    p = Path(path) if path is not None else Path(config.RAW_DATA_PATH)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


def save_raw_snapshot(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Overwrite the raw parquet snapshot. Returns the path written."""
    out = _resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    to_write = df.sort_values("timestamp").reset_index(drop=True)
    to_write.to_parquet(out, index=False)
    print(
        f"[raw_io] Saved {len(to_write)} rows to {out} "
        f"({to_write['timestamp'].min()} -> {to_write['timestamp'].max()})"
    )
    return out


def load_raw_snapshot(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw parquet snapshot. Raises FileNotFoundError if missing."""
    out = _resolve_path(path)
    if not out.exists():
        raise FileNotFoundError(
            f"Raw snapshot not found at {out}. "
            f"Run: python -m src.feature_pipeline raw-snapshot"
        )
    df = pd.read_parquet(out)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df.sort_values("timestamp").reset_index(drop=True)


def upsert_raw_snapshot(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """
    Merge `df` into the existing raw snapshot on `timestamp` (keep last),
    then rewrite. Creates the file if it does not exist yet.
    """
    out = _resolve_path(path)
    incoming = df.copy()
    incoming["timestamp"] = pd.to_datetime(incoming["timestamp"]).dt.tz_localize(None)

    if out.exists():
        existing = load_raw_snapshot(out)
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming

    combined = (
        combined.drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return save_raw_snapshot(combined, out)
