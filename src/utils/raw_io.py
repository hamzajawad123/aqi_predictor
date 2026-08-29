"""Save and load the raw pollution + weather parquet."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config

# Always the project root, even if you run from notebooks/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path | None = None) -> Path:
    p = Path(path) if path is not None else Path(config.RAW_DATA_PATH)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


def save_raw_snapshot(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Write the raw parquet. Returns the path."""
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
    """Read the raw parquet."""
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
    """Add new hours into the existing parquet (keep last if duplicate)."""
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
