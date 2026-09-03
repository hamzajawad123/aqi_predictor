"""Prophet on AQI level (not the change). Fit on train+val so it is not asked to guess too far ahead."""
from __future__ import annotations

import logging
import os
import shutil

import pandas as pd
from prophet import Prophet

from src.utils.evaluation import evaluate

_CMDSTAN_READY = False


def _ensure_cmdstan_for_prophet() -> bool:
    """Install CmdStan and point Prophet 1.1.6 at it (same steps as the training notebook)."""
    global _CMDSTAN_READY
    if _CMDSTAN_READY:
        return True
    try:
        import cmdstanpy
        import prophet

        try:
            working_cmdstan_path = cmdstanpy.cmdstan_path()
        except Exception:
            print("[train_prophet] Installing CmdStan (this can take several minutes)...")
            cmdstanpy.install_cmdstan()
            working_cmdstan_path = cmdstanpy.cmdstan_path()

        prophet_dir = os.path.dirname(prophet.__file__)
        expected_cmdstan_dir = os.path.join(prophet_dir, "stan_model", "cmdstan-2.33.1")
        if os.path.islink(expected_cmdstan_dir) or os.path.exists(expected_cmdstan_dir):
            if os.path.islink(expected_cmdstan_dir):
                os.unlink(expected_cmdstan_dir)
            else:
                shutil.rmtree(expected_cmdstan_dir)
        os.symlink(working_cmdstan_path, expected_cmdstan_dir)
        _CMDSTAN_READY = True
        print("[train_prophet] Linked CmdStan to Prophet.")
        return True
    except Exception as e:
        print(f"[train_prophet] CmdStan setup failed ({type(e).__name__}: {e})")
        return False


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

    if not _ensure_cmdstan_for_prophet():
        print("[train_prophet] Skipping Prophet.")
        return None, None

    try:
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )
        model.fit(prophet_train)
    except Exception as e:
        print(f"[train_prophet] Skipping Prophet ({type(e).__name__}: {e})")
        return None, None

    future = pd.DataFrame({
        "ds": test_df["timestamp"] + pd.Timedelta(hours=horizon_hours)
    })
    y_pred = model.predict(future)["yhat"].values

    return model, evaluate(test_df[target_col].values, y_pred, report_name)
