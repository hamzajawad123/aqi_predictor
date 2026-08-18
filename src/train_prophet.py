"""
Prophet model for AQI forecasting — the classical statistical modelling
entry in the comparison table.

WHY THIS WAS ADDED: the project brief asks for "a variety of forecasting
models, from statistical modelling to deep learning." Ridge/RandomForest/
XGBoost/LightGBM are regression models applied to engineered features — not
classical time-series decomposition. Prophet (trend + seasonality
decomposition) is the actual statistical end of that spectrum, so it closes
a real gap rather than just adding a 7th model for its own sake.

STAYS ON ABSOLUTE AQI, not the delta target: Prophet models a level series as
trend + seasonality. A mean-reverting delta has no trend to decompose, so
delta-framing it would strip out the only thing Prophet contributes. It is
the one model in the comparison that is not reframed, by design.

FIT THROUGH THE FORECAST ORIGIN: Prophet must be fitted on history up to the
point it forecasts from, then asked for at most `horizon_hours` ahead — that
is how it would run in deployment. Fitting on train only and then predicting
test timestamps a year or more later makes it extrapolate its linear trend far
past any data, which on Lahore's steeply declining AQI produced predictions
thousands of units off (RMSE ~2100 on a 0-1000 scale). Pass fit_df =
train + val so the gap between the fit end and the test period is closed.

DELIBERATELY UNIVARIATE — no weather regressors: a fair comparison against
the other models requires using only information available at "now" (time t)
to forecast t+h. Prophet's own regressor mechanism needs KNOWN future values
of any regressor at prediction time — feeding it the *actual* future wind
speed/humidity would leak information no real deployment would have. So this
fits purely on AQI's own trend + daily/weekly/yearly seasonality.

NOT eligible for the Model Registry / API serving (same reasoning as LSTM/
GRU in training_pipeline.py): Prophet's interface takes a `ds` (date) column,
not a flat feature-row like api/main.py's /predict endpoint expects. Compared
here, not served.
"""
from __future__ import annotations

import logging

import pandas as pd
from prophet import Prophet

from src.utils.evaluation import evaluate


def train_prophet_model(fit_df: pd.DataFrame, test_df: pd.DataFrame,
                        target_col: str = "aqi_target_72h",
                        horizon_hours: int = 72,
                        report_name: str = "Prophet"):
    """
    Fit Prophet on fit_df's own (timestamp, aqi) history, then forecast AQI at
    each test row's TARGET timestamp (that row's timestamp + horizon_hours) —
    which is exactly what aqi_target_{h}h represents, so predictions line up
    with actuals without extra alignment.

    fit_df should run right up to the start of the test period (i.e. train +
    val), otherwise Prophet extrapolates its trend across the unseen gap.
    """
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    prophet_train = (
        fit_df[["timestamp", "aqi"]]
        .rename(columns={"timestamp": "ds", "aqi": "y"})
        .dropna()
        .drop_duplicates(subset="ds")
        .sort_values("ds")
        .reset_index(drop=True)
    )

    fit_end = prophet_train["ds"].max()
    forecast_start = test_df["timestamp"].min() + pd.Timedelta(hours=horizon_hours)
    lead_days = (forecast_start - fit_end).total_seconds() / 86400
    if lead_days > 7:
        print(
            f"[train_prophet] WARNING: first forecast point is {lead_days:.0f} days "
            f"past the fit end ({fit_end.date()}) — Prophet is extrapolating its "
            f"trend, results will not be comparable. Pass train+val as fit_df."
        )

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    model.fit(prophet_train)

    future = pd.DataFrame({
        "ds": test_df["timestamp"] + pd.Timedelta(hours=horizon_hours)
    })
    y_pred = model.predict(future)["yhat"].values

    return model, evaluate(test_df[target_col].values, y_pred, report_name)
