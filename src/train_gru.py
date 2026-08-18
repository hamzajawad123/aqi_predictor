"""
GRU model for AQI forecasting (TensorFlow/Keras).

Lighter-weight recurrent alternative to the LSTM. Both share one
implementation in train_lstm.train_recurrent_model so the delta-reconstruction
and scaling logic can't drift apart between the two.
"""
from __future__ import annotations

from src.train_lstm import (  # re-exported for callers that import from here
    SEQUENCE_LENGTH,
    build_recurrent,
    make_sequences,
    train_recurrent_model,
)


def build_gru(n_timesteps: int, n_features: int):
    return build_recurrent("GRU", n_timesteps, n_features)


def train_gru_model(train_df, val_df, test_df, feature_cols, target_col,
                    absolute_target_col: str | None = None,
                    report_name: str = "GRU", **kwargs):
    return train_recurrent_model(
        "GRU", train_df, val_df, test_df, feature_cols, target_col,
        absolute_target_col=absolute_target_col,
        report_name=report_name,
        plot_path=kwargs.pop("plot_path", "reports/gru_loss_curve.png"),
        **kwargs,
    )
