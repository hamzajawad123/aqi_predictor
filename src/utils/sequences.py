"""Turn a table into 24-hour windows for LSTM / GRU."""
from __future__ import annotations

import numpy as np

SEQUENCE_LENGTH = 24


def make_sequences(
    df,
    feature_cols: list[str],
    target_col: str,
    seq_len: int = SEQUENCE_LENGTH,
    anchor: np.ndarray | None = None,
    absolute_target: np.ndarray | None = None,
):
    """Each window ends on the hour we predict from. Pass unscaled AQI as anchor."""
    values = np.asarray(df[feature_cols].values, dtype=float)
    targets = np.asarray(df[target_col].values, dtype=float)
    n = len(df)
    if n < seq_len:
        raise ValueError(f"Need at least {seq_len} rows to build sequences, got {n}.")

    end_idx = np.arange(seq_len - 1, n)
    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=seq_len, axis=0
    )  # (n - seq_len + 1, n_features, seq_len)
    X = np.ascontiguousarray(windows.transpose(0, 2, 1))
    y = targets[end_idx]

    anchor_out = None if anchor is None else np.asarray(anchor, dtype=float)[end_idx]
    abs_out = (
        None if absolute_target is None
        else np.asarray(absolute_target, dtype=float)[end_idx]
    )
    return X, y, anchor_out, abs_out
