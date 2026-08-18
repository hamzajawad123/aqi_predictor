"""
Recurrent models (LSTM / GRU) for AQI forecasting — TensorFlow/Keras.

Kept in its own module because recurrent nets need a windowed 3D input
(samples, timesteps, features) instead of the flat 2D table the
scikit-learn / XGBoost models use. train_gru.py delegates here so the two
share one implementation.

When target_col is an aqi_delta_* column, metrics are reported on
reconstructed absolute AQI (aqi at the prediction origin + predicted delta)
vs the matching aqi_target_* column.

Windowing (and the two alignment rules that broke a whole run before) lives in
src/utils/sequences.py so it can be tested without importing TensorFlow.
"""
from __future__ import annotations

import os

from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from src.utils.evaluation import evaluate, reconstruct_absolute
from src.utils.sequences import SEQUENCE_LENGTH, make_sequences  # noqa: F401 (re-exported)


def build_recurrent(kind: str, n_timesteps: int, n_features: int) -> keras.Model:
    """Two stacked recurrent layers + dense head. kind is 'LSTM' or 'GRU'."""
    layer = keras.layers.LSTM if kind.upper() == "LSTM" else keras.layers.GRU
    model = keras.Sequential([
        keras.layers.Input(shape=(n_timesteps, n_features)),
        layer(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        layer(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model


# Kept for backwards compatibility with older imports
def build_lstm(n_timesteps: int, n_features: int) -> keras.Model:
    return build_recurrent("LSTM", n_timesteps, n_features)


def train_recurrent_model(
    kind: str,
    train_df,
    val_df,
    test_df,
    feature_cols: list[str],
    target_col: str,
    absolute_target_col: str | None = None,
    anchor_col: str = "aqi",
    report_name: str | None = None,
    scale_target: bool = True,
    epochs: int = 100,
    batch_size: int = 64,
    verbose: int = 0,
    plot_path: str | None = None,
):
    """
    Train one recurrent model on (usually) the delta target and score it on
    reconstructed absolute AQI.

    The delta target is standardized before the MSE loss sees it (features
    already are), which keeps gradient scales comparable and stops the net
    from spending its early epochs just learning the output magnitude.

    Returns (model, metrics, meta). meta carries everything needed to fit a
    shrinkage factor on validation and to serve the model later.
    """
    name = report_name or kind.upper()

    # Anchors and absolute truths are read BEFORE any scaling touches the frame.
    anchors = {
        "train": train_df[anchor_col].to_numpy(dtype=float),
        "val": val_df[anchor_col].to_numpy(dtype=float),
        "test": test_df[anchor_col].to_numpy(dtype=float),
    }
    abs_truth = {"train": None, "val": None, "test": None}
    if absolute_target_col and absolute_target_col in test_df.columns:
        abs_truth = {
            "train": train_df[absolute_target_col].to_numpy(dtype=float),
            "val": val_df[absolute_target_col].to_numpy(dtype=float),
            "test": test_df[absolute_target_col].to_numpy(dtype=float),
        }

    feature_scaler = StandardScaler().fit(train_df[feature_cols])
    tr, va, te = train_df.copy(), val_df.copy(), test_df.copy()
    for frame in (tr, va, te):
        frame[feature_cols] = feature_scaler.transform(frame[feature_cols])

    X_train, y_train, _, _ = make_sequences(tr, feature_cols, target_col)
    X_val, y_val, val_anchor, val_abs = make_sequences(
        va, feature_cols, target_col,
        anchor=anchors["val"], absolute_target=abs_truth["val"],
    )
    X_test, y_test, test_anchor, test_abs = make_sequences(
        te, feature_cols, target_col,
        anchor=anchors["test"], absolute_target=abs_truth["test"],
    )

    target_scaler = None
    y_train_fit, y_val_fit = y_train, y_val
    if scale_target:
        target_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train_fit = target_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_val_fit = target_scaler.transform(y_val.reshape(-1, 1)).ravel()

    model = build_recurrent(kind, SEQUENCE_LENGTH, len(feature_cols))
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train_fit,
        validation_data=(X_val, y_val_fit),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )

    def _predict_delta(X):
        raw = model.predict(X, batch_size=256, verbose=0).reshape(-1, 1)
        if target_scaler is not None:
            raw = target_scaler.inverse_transform(raw)
        return raw.ravel()

    val_delta_pred = _predict_delta(X_val)
    test_delta_pred = _predict_delta(X_test)

    if test_abs is not None and test_anchor is not None:
        result = evaluate(
            test_abs, reconstruct_absolute(test_anchor, test_delta_pred), name
        )
    else:
        result = evaluate(y_test, test_delta_pred, name)

    if plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
        plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss (MSE)"); plt.legend()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

    meta = {
        "scaler": feature_scaler,
        "target_scaler": target_scaler,
        "sequence_length": SEQUENCE_LENGTH,
        "val_delta_pred": val_delta_pred,
        "val_anchor": val_anchor,
        "val_absolute_true": val_abs,
        "test_delta_pred": test_delta_pred,
        "test_anchor": test_anchor,
        "test_absolute_true": test_abs,
    }
    return model, result, meta


def train_lstm_model(train_df, val_df, test_df, feature_cols, target_col,
                     absolute_target_col: str | None = None,
                     report_name: str = "LSTM", **kwargs):
    return train_recurrent_model(
        "LSTM", train_df, val_df, test_df, feature_cols, target_col,
        absolute_target_col=absolute_target_col,
        report_name=report_name,
        plot_path=kwargs.pop("plot_path", "reports/lstm_loss_curve.png"),
        **kwargs,
    )
