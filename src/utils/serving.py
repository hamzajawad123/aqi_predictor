"""Load models and make forecasts. Used by FastAPI and Streamlit."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src import config
from src.utils.evaluation import reconstruct_absolute
from src.utils.hopsworks_utils import get_feature_store, get_model_registry

HAZARDOUS_THRESHOLD = 151
_TTL_SEC = 300.0
_HW_LOCK = threading.Lock()
_MODELS: dict[int, Any] | None = None
_FRAME: pd.DataFrame | None = None
_FRAME_AT = 0.0
_CACHE_ROOT = Path(__file__).resolve().parents[2] / ".hw_cache" / "models"

_LOG_POLLUTANTS = ("pm2_5", "pm10", "o3", "no2", "so2", "co", "no", "nh3")
_WEATHER = ("temperature", "humidity", "pressure", "wind_speed")


def _drop_cols(columns) -> set[str]:
    drop = {"timestamp", "hour", "month", "openweather_aqi_category"}
    for h in config.TARGET_HORIZONS:
        drop.add(f"aqi_target_{h}h")
        drop.add(f"aqi_delta_{h}h")
    return drop


def _joblib_in(folder: Path, name: str) -> Path | None:
    direct = folder / f"{name}.joblib"
    if direct.is_file():
        return direct
    matches = list(folder.rglob(f"{name}.joblib"))
    return matches[0] if matches else None


def _download_payload(hw_model, name: str):
    dest = _CACHE_ROOT / name / str(hw_model.version)
    dest.mkdir(parents=True, exist_ok=True)
    existing = _joblib_in(dest, name)
    if existing is not None:
        return joblib.load(existing)
    model_dir = hw_model.download(local_path=str(dest))
    payload_path = _joblib_in(Path(model_dir), name) or _joblib_in(dest, name)
    if payload_path is None:
        raise FileNotFoundError(f"No {name}.joblib under {model_dir} or {dest}")
    return joblib.load(payload_path)


def load_models() -> dict[int, Any]:
    global _MODELS
    with _HW_LOCK:
        if _MODELS is not None:
            return _MODELS
        mr = get_model_registry()
        models: dict[int, Any] = {}
        for h in config.TARGET_HORIZONS:
            name = config.model_name_for_horizon(h)
            try:
                hw_model = mr.get_model(name)
                payload = _download_payload(hw_model, name)
                metrics = {}
                try:
                    metrics = dict(hw_model.training_metrics or {})
                except Exception:
                    pass
                models[h] = {
                    "payload": payload,
                    "version": hw_model.version,
                    "name": name,
                    "metrics": metrics,
                }
            except Exception as e:
                models[h] = {"error": str(e), "name": name, "metrics": {}}
        _MODELS = models
        return _MODELS


def load_feature_frame() -> pd.DataFrame:
    global _FRAME, _FRAME_AT
    now = time.time()
    with _HW_LOCK:
        if _FRAME is not None and (now - _FRAME_AT) < _TTL_SEC:
            return _FRAME
        fs = get_feature_store()
        fg = fs.get_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
        )
        df = fg.read()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        _FRAME = df
        _FRAME_AT = now
        return _FRAME


def _predict_one(payload: dict, latest_row, feature_fallback: list[str]) -> float:
    model_obj = payload["model"]
    feature_cols = payload.get("feature_cols") or feature_fallback
    target_type = payload.get("target_type", "delta")
    shrinkage = float(payload.get("shrinkage", 1.0))
    aqi_now = float(latest_row["aqi"])
    x = latest_row[feature_cols].values.reshape(1, -1)

    if isinstance(model_obj, dict) and model_obj.get("type") == "mean_ensemble":
        preds = []
        for member in model_obj["members"].values():
            if isinstance(member, dict):
                preds.append(
                    float(member["model"].predict(x)[0])
                    * float(member.get("shrinkage", 1.0))
                )
            else:
                preds.append(float(member.predict(x)[0]))
        delta_or_abs = float(np.mean(preds))
    else:
        delta_or_abs = float(model_obj.predict(x)[0])

    if target_type == "delta":
        return float(reconstruct_absolute(aqi_now, delta_or_abs, shrinkage))
    return delta_or_abs


def predict_forecasts(df: pd.DataFrame | None = None) -> list[dict]:
    df = load_feature_frame() if df is None else df
    models = load_models()
    latest_row = df.iloc[-1]
    feature_fallback = [c for c in df.columns if c not in _drop_cols(df.columns)]
    forecasts: list[dict] = []
    for h in config.TARGET_HORIZONS:
        day = h // 24
        entry = models.get(h, {})
        metrics = entry.get("metrics") or {}
        rmse = metrics.get("RMSE") or metrics.get("rmse")
        if "error" in entry or "payload" not in entry:
            forecasts.append(
                {
                    "horizon_hours": h,
                    "day": day,
                    "predicted_aqi": None,
                    "model_used": entry.get("name"),
                    "status": "unavailable",
                    "detail": entry.get("error", "No registered model"),
                    "rmse": float(rmse) if rmse is not None else None,
                }
            )
            continue
        try:
            pred = _predict_one(entry["payload"], latest_row, feature_fallback)
            forecasts.append(
                {
                    "horizon_hours": h,
                    "day": day,
                    "predicted_aqi": pred,
                    "model_used": f"{entry['name']}@v{entry['version']}",
                    "status": "ok",
                    "detail": None,
                    "rmse": float(rmse) if rmse is not None else None,
                }
            )
        except Exception as e:
            forecasts.append(
                {
                    "horizon_hours": h,
                    "day": day,
                    "predicted_aqi": None,
                    "model_used": entry.get("name"),
                    "status": "error",
                    "detail": str(e),
                    "rmse": float(rmse) if rmse is not None else None,
                }
            )
    return forecasts


def predict_payload() -> dict:
    """JSON body for GET /predict."""
    df = load_feature_frame()
    forecasts = predict_forecasts(df)
    ok = [f["predicted_aqi"] for f in forecasts if f.get("predicted_aqi") is not None]
    return {
        "city": config.CITY_NAME,
        "forecasts": forecasts,
        "forecast_72h": [
            {"horizon_hours": f["horizon_hours"], "predicted_aqi": f["predicted_aqi"]}
            for f in forecasts
            if f.get("predicted_aqi") is not None
        ],
        "hazardous_alert": any(p >= HAZARDOUS_THRESHOLD for p in ok),
        "current_aqi": float(df.iloc[-1]["aqi"]),
    }


def _maybe_expm1(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    if float(s.quantile(0.95)) < 15:
        return np.expm1(s.clip(lower=0))
    return s


def _invert_pollutants(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in _LOG_POLLUTANTS:
        if col in out.columns:
            out[col] = _maybe_expm1(out[col])
    return out


def _pkt(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("Asia/Karachi")


def dashboard_state() -> dict:
    """One payload for the Streamlit page."""
    df = load_feature_frame()
    forecasts = predict_forecasts(df)
    latest = df.iloc[-1]
    raw = _invert_pollutants(df)
    latest_raw = raw.iloc[-1]
    ok = [f["predicted_aqi"] for f in forecasts if f.get("predicted_aqi") is not None]

    pollutants = {}
    for col in ("pm2_5", "pm10", "o3", "no2", "so2", "co"):
        if col in latest_raw.index and pd.notna(latest_raw[col]):
            pollutants[col] = float(latest_raw[col])

    weather = {}
    for col in _WEATHER:
        if col in latest.index and pd.notna(latest[col]):
            weather[col] = float(latest[col])

    hist_24 = raw.tail(24)[["timestamp", "aqi"] + [c for c in ("pm2_5",) if c in raw.columns]].copy()
    daily = (
        raw.set_index("timestamp")["aqi"]
        .resample("D")
        .mean()
        .dropna()
        .rename("aqi")
        .reset_index()
    )
    recent = raw.tail(90 * 24).copy()

    lookback = raw.tail(30 * 24)
    drivers = []
    driver_cols = [
        ("pm2_5", "PM2.5"),
        ("pm10", "PM10"),
        ("no2", "NO2"),
        ("o3", "O3"),
        ("wind_speed", "Wind speed"),
        ("humidity", "Humidity"),
        ("temperature", "Temperature"),
        ("aqi", "AQI"),
    ]
    for col, label in driver_cols:
        if col not in lookback.columns:
            continue
        now_v = pd.to_numeric(latest_raw[col], errors="coerce")
        med = pd.to_numeric(lookback[col], errors="coerce").median()
        if pd.isna(now_v) or pd.isna(med) or med == 0:
            continue
        drivers.append(
            {
                "feature": label,
                "now": float(now_v),
                "median": float(med),
                "delta_pct": float((now_v - med) / abs(med) * 100.0),
            }
        )
    drivers.sort(key=lambda d: abs(d["delta_pct"]), reverse=True)

    aqi_24h_ago = float(hist_24.iloc[0]["aqi"]) if len(hist_24) else None

    return {
        "city": config.CITY_NAME,
        "current_aqi": float(latest["aqi"]),
        "aqi_24h_ago": aqi_24h_ago,
        "timestamp": _pkt(latest["timestamp"]),
        "forecasts": forecasts,
        "hazardous_alert": any(p >= HAZARDOUS_THRESHOLD for p in ok),
        "pollutants": pollutants,
        "weather": weather,
        "history_24h": hist_24,
        "history_daily": daily,
        "recent": recent,
        "drivers": drivers,
    }
