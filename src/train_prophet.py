"""Prophet on AQI level (not the change). Fit on train+val so it is not asked to guess too far ahead."""
from __future__ import annotations

import logging

import pandas as pd
from prophet import Prophet

from src.utils.evaluation import evaluate


def train_prophet_model(fit_df: pd.DataFrame, test_df: pd.DataFrame,
                        target_col: str = "aqi_target_72h",
                        horizon_hours: int = 72,
                        report_name: str = "Prophet"):
    """Fit Prophet, then predict AQI at each test time plus the horizon."""
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
