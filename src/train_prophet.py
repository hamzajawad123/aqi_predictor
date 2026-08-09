"""
Prophet model for AQI forecasting — the classical statistical modelling
entry in the comparison table.

WHY THIS WAS ADDED: the project brief asks for "a variety of forecasting
models, from statistical modelling to deep learning." Ridge/RandomForest/
XGBoost/LightGBM are regression models applied to engineered features — not
classical time-series decomposition. Prophet (trend + seasonality
decomposition) is the actual statistical end of that spectrum, so it closes
a real gap rather than just adding a 7th model for its own sake.

DELIBERATELY UNIVARIATE — no weather regressors: a fair comparison against
the other models requires using only information available at "now" (time t)
to forecast t+72h. Prophet's own regressor mechanism needs KNOWN future
values of any regressor at prediction time — feeding it the *actual* future
wind speed/humidity would leak information no real deployment would have at
prediction time. So this fits purely on AQI's own trend + daily/weekly/yearly
seasonality, which is the honest, leak-free version of "what can a classical
statistical model do with just the target's own history".

NOT eligible for the Model Registry / API serving (same reasoning as LSTM/
GRU in training_pipeline.py): Prophet's interface takes a `ds` (date) column,
not a flat feature-row like api/main.py's /predict endpoint expects. Compared
here, not served.
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_prophet_model(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         target_col: str = "aqi_target_72h",
                         horizon_hours: int = 72):
    """
    Fits Prophet on the train period's own (timestamp, aqi) history, then
    asks it to forecast AQI at each test row's TARGET timestamp
    (that row's timestamp + horizon_hours) — which is exactly what
    `aqi_target_72h` already represents, so predictions line up with
    actuals with no extra alignment needed.
    """
    prophet_train = train_df[["timestamp", "aqi"]].rename(
        columns={"timestamp": "ds", "aqi": "y"}
    )

    print("=" * 80)
    print("Shape:", prophet_train.shape)

    print("\nColumns:")
    print(prophet_train.columns.tolist())

    print("\nDuplicate column names:")
    print(prophet_train.columns[prophet_train.columns.duplicated()])

    print("\nDuplicate ds values:")
    print(prophet_train["ds"].duplicated().sum())

    print("\nDuplicate index:")
    print(not prophet_train.index.is_unique)

    print("\nMissing ds:")
    print(prophet_train["ds"].isna().sum())

    print("\nMissing y:")
    print(prophet_train["y"].isna().sum())

    print("=" * 80)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    # Suppress Prophet/cmdstanpy's verbose fitting logs
    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    model.fit(prophet_train)

    future = pd.DataFrame({
        "ds": test_df["timestamp"] + pd.Timedelta(hours=horizon_hours)
    })
    forecast = model.predict(future)

    y_pred = forecast["yhat"].values
    y_true = test_df[target_col].values

    result = {
        "model": "Prophet",
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
    return model, result
