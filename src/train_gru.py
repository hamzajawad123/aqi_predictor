"""
GRU model for AQI forecasting (TensorFlow/Keras).
Structurally identical to train_lstm.py (same windowing, same scaling, same
EarlyStopping approach) — GRU is a lighter-weight recurrent alternative to
LSTM (fewer gates, fewer parameters), included so the model comparison table
covers both rather than assuming one is better without checking.
"""
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.train_lstm import make_sequences, SEQUENCE_LENGTH


def build_gru(n_timesteps: int, n_features: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(n_timesteps, n_features)),
        keras.layers.GRU(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.GRU(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model


def train_gru_model(train_df, val_df, test_df, feature_cols, target_col):
    scaler = StandardScaler().fit(train_df[feature_cols])
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    X_train, y_train = make_sequences(train_df, feature_cols, target_col)
    X_val, y_val = make_sequences(val_df, feature_cols, target_col)
    X_test, y_test = make_sequences(test_df, feature_cols, target_col)

    model = build_gru(SEQUENCE_LENGTH, len(feature_cols))

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    y_pred = model.predict(X_test).flatten()
    result = {
        "model": "GRU",
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred)),
    }

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss (MSE)"); plt.legend()
    plt.savefig("reports/gru_loss_curve.png", bbox_inches="tight")
    plt.close()

    return model, result
