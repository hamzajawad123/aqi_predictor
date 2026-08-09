"""
LSTM model for AQI forecasting (TensorFlow/Keras).
Kept in its own module because LSTMs need a windowed 3D input
(samples, timesteps, features) instead of the flat 2D table the
scikit-learn / XGBoost models use.

Called from training_pipeline.py once the tabular models are working.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEQUENCE_LENGTH = 24  # look back 24 hours to predict the target horizon


def make_sequences(df, feature_cols, target_col, seq_len=SEQUENCE_LENGTH):
    """Turn a flat feature table into (samples, timesteps, features) windows."""
    X, y = [], []
    values = df[feature_cols].values
    targets = df[target_col].values
    for i in range(seq_len, len(df)):
        X.append(values[i - seq_len:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def build_lstm(n_timesteps: int, n_features: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(n_timesteps, n_features)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model


def train_lstm_model(train_df, val_df, test_df, feature_cols, target_col):
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

    model = build_lstm(SEQUENCE_LENGTH, len(feature_cols))

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,           # high ceiling; early stopping decides the real number
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    y_pred = model.predict(X_test).flatten()
    result = {
        "model": "LSTM",
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred)),
    }

    # Save the train/val loss curve for the report — shows over/underfitting was checked.
    import matplotlib.pyplot as plt
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss (MSE)"); plt.legend()
    plt.savefig("reports/lstm_loss_curve.png", bbox_inches="tight")

    return model, result
