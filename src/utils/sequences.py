"""
Windowing helper for the recurrent models.

Lives outside train_lstm.py on purpose: the alignment rules below are where
the recurrent path silently broke before, and keeping them free of the
TensorFlow import means they stay unit-testable anywhere (TF is only installed
for the Colab/GPU training runs).
"""
from __future__ import annotations

import numpy as np

SEQUENCE_LENGTH = 24  # look back 24 hours to predict the target horizon


def make_sequences(
    df,
    feature_cols: list[str],
    target_col: str,
    seq_len: int = SEQUENCE_LENGTH,
    anchor: np.ndarray | None = None,
    absolute_target: np.ndarray | None = None,
):
    """
    Turn a flat feature table into (samples, timesteps, features) windows.

    Window i covers rows [i-seq_len+1 .. i] INCLUSIVE and predicts the target at
    row i. Ending the window on the prediction origin matters for delta targets:
    the delta is defined relative to the AQI at row i, so a window stopping at
    i-1 asks the net for a delta against a value it was never shown, while the
    tabular models do get row i.

    `anchor` (current AQI) and `absolute_target` (absolute future AQI) are
    optional row-aligned arrays sliced the same way as the targets. They must be
    UNSCALED — `aqi` is itself a model feature, so scaling the frame in place
    standardizes it, and reconstructing from that adds ~0 instead of the real
    AQI level, silently turning the absolute prediction into the bare delta.
    Passing them in explicitly is what keeps that impossible.

    Returns (X, y, anchor_windowed, absolute_target_windowed); the last two are
    None when not supplied.
    """
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
