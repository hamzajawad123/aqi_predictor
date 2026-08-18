"""
Inference Pipeline (served as a FastAPI app)
=============================================
- Loads per-horizon winners from the Hopsworks Model Registry
  (aqi_forecaster_24h / _48h / _72h).
- Loads the latest features from the Hopsworks Feature Store.
- Returns day-1 / day-2 / day-3 AQI forecasts (delta models reconstruct
  absolute AQI as current_aqi + predicted_delta).

Consumed by the Streamlit dashboard (app/) — NOT called directly by end users.
"""
from __future__ import annotations

import functools
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src import config
from src.utils.evaluation import reconstruct_absolute
from src.utils.hopsworks_utils import get_feature_store, get_model_registry

app = FastAPI(title="AQI Predictor API", version="2.0.0")

# Continuous US EPA AQI — Unhealthy for Sensitive Groups and worse
HAZARDOUS_THRESHOLD = 151


class HorizonForecast(BaseModel):
    horizon_hours: int
    day: int
    predicted_aqi: float | None = None
    model_used: str | None = None
    status: str = "ok"
    detail: str | None = None


class ForecastResponse(BaseModel):
    city: str
    forecasts: list[HorizonForecast]
    # Backward-compatible alias used by older Streamlit pages
    forecast_72h: list[dict]
    hazardous_alert: bool
    current_aqi: float | None = None


def _drop_cols(columns) -> set[str]:
    drop = {"timestamp", "hour", "month", "openweather_aqi_category"}
    for h in config.TARGET_HORIZONS:
        drop.add(f"aqi_target_{h}h")
        drop.add(f"aqi_delta_{h}h")
    return drop


@functools.lru_cache(maxsize=1)
def _load_models() -> dict[int, Any]:
    """Load one registered model payload per horizon (cached)."""
    mr = get_model_registry()
    models = {}
    for h in config.TARGET_HORIZONS:
        name = config.model_name_for_horizon(h)
        try:
            hw_model = mr.get_model(name)
            model_dir = hw_model.download()
            payload = joblib.load(f"{model_dir}/{name}.joblib")
            models[h] = {"payload": payload, "version": hw_model.version, "name": name}
        except Exception as e:
            models[h] = {"error": str(e), "name": name}
    return models


def _predict_one(payload: dict, latest_row, feature_fallback: list[str]) -> float:
    """
    Run one horizon model; reconstruct absolute AQI if target_type is delta.

    The shrinkage factor stored at registration time is applied here too —
    training scored the model as aqi + shrinkage * delta, so serving the raw
    delta instead would silently not be the model that passed the gate.
    """
    model_obj = payload["model"]
    feature_cols = payload.get("feature_cols") or feature_fallback
    target_type = payload.get("target_type", "delta")
    shrinkage = float(payload.get("shrinkage", 1.0))
    aqi_now = float(latest_row["aqi"])
    x = latest_row[feature_cols].values.reshape(1, -1)

    if isinstance(model_obj, dict) and model_obj.get("type") == "mean_ensemble":
        preds = []
        for member in model_obj["members"].values():
            # Members registered after the shrinkage change carry their own factor
            if isinstance(member, dict):
                preds.append(float(member["model"].predict(x)[0])
                             * float(member.get("shrinkage", 1.0)))
            else:
                preds.append(float(member.predict(x)[0]))
        delta_or_abs = float(np.mean(preds))
    else:
        delta_or_abs = float(model_obj.predict(x)[0])

    if target_type == "delta":
        return float(reconstruct_absolute(aqi_now, delta_or_abs, shrinkage))
    return delta_or_abs


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict", response_model=ForecastResponse)
def predict():
    models = _load_models()

    fs = get_feature_store()
    fg = fs.get_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
    )
    recent_df = fg.read()
    latest_row = recent_df.sort_values("timestamp").iloc[-1]
    feature_fallback = [c for c in recent_df.columns if c not in _drop_cols(recent_df.columns)]

    forecasts: list[HorizonForecast] = []
    for h in config.TARGET_HORIZONS:
        day = h // 24
        entry = models.get(h, {})
        if "error" in entry or "payload" not in entry:
            forecasts.append(
                HorizonForecast(
                    horizon_hours=h,
                    day=day,
                    status="unavailable",
                    detail=entry.get("error", "No registered model for this horizon"),
                    model_used=entry.get("name"),
                )
            )
            continue
        try:
            pred = _predict_one(entry["payload"], latest_row, feature_fallback)
            forecasts.append(
                HorizonForecast(
                    horizon_hours=h,
                    day=day,
                    predicted_aqi=pred,
                    model_used=f"{entry['name']}@v{entry['version']}",
                    status="ok",
                )
            )
        except Exception as e:
            forecasts.append(
                HorizonForecast(
                    horizon_hours=h,
                    day=day,
                    status="error",
                    detail=str(e),
                    model_used=entry.get("name"),
                )
            )

    ok_preds = [f.predicted_aqi for f in forecasts if f.predicted_aqi is not None]
    hazardous = any(p >= HAZARDOUS_THRESHOLD for p in ok_preds)

    # Legacy field for older Streamlit pages
    legacy = [
        {"horizon_hours": f.horizon_hours, "predicted_aqi": f.predicted_aqi}
        for f in forecasts
        if f.predicted_aqi is not None
    ]

    return ForecastResponse(
        city=config.CITY_NAME,
        forecasts=forecasts,
        forecast_72h=legacy,
        hazardous_alert=bool(hazardous),
        current_aqi=float(latest_row["aqi"]),
    )


@app.get("/model-metrics")
def model_metrics():
    """Per-horizon training metrics from the registry."""
    mr = get_model_registry()
    out = {}
    for h in config.TARGET_HORIZONS:
        name = config.model_name_for_horizon(h)
        try:
            hw_model = mr.get_model(name)
            out[name] = hw_model.training_metrics
        except Exception as e:
            out[name] = {"error": str(e)}
    return out
