"""
Inference Pipeline (served as a FastAPI app)
=============================================
Thin HTTP wrapper around src.utils.serving — the Streamlit dashboard can
call the same serving module directly (needed for Streamlit Cloud).
"""
from __future__ import annotations

import threading

from fastapi import FastAPI
from pydantic import BaseModel

from src.utils import serving

app = FastAPI(title="AQI Predictor API", version="2.0.0")


class HorizonForecast(BaseModel):
    horizon_hours: int
    day: int
    predicted_aqi: float | None = None
    model_used: str | None = None
    status: str = "ok"
    detail: str | None = None
    rmse: float | None = None


class ForecastResponse(BaseModel):
    city: str
    forecasts: list[HorizonForecast]
    forecast_72h: list[dict]
    hazardous_alert: bool
    current_aqi: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


def _warmup() -> None:
    try:
        serving.load_models()
        serving.load_feature_frame()
        serving.predict_payload()
    except Exception:
        pass


@app.on_event("startup")
def _warmup_models() -> None:
    threading.Thread(target=_warmup, daemon=True, name="warmup-models").start()


@app.get("/predict", response_model=ForecastResponse)
def predict():
    return serving.predict_payload()


@app.get("/model-metrics")
def model_metrics():
    models = serving.load_models()
    return {
        entry.get("name", str(h)): entry.get("metrics") or entry.get("error")
        for h, entry in models.items()
    }
