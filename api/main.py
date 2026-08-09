"""
Inference Pipeline (served as a FastAPI app)
=============================================
- Loads the current best model from the Hopsworks Model Registry.
- Loads the latest features from the Hopsworks Feature Store.
- Produces a 3-day AQI forecast.
- Flags hazardous AQI levels.

Consumed by the Streamlit dashboard (app/) — NOT called directly by end users.
"""
import functools
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from src import config
from src.utils.hopsworks_utils import get_feature_store, get_model_registry

app = FastAPI(title="AQI Predictor API", version="1.0.0")

# WHO/EPA-style hazardous thresholds — adjust to whichever AQI scale your
# OpenWeather response uses (OpenWeather's own 1-5 scale, or US AQI 0-500).
HAZARDOUS_THRESHOLD = 4  # OpenWeather scale: 1=Good ... 5=Very Poor


class ForecastResponse(BaseModel):
    city: str
    forecast_72h: list[dict]
    hazardous_alert: bool
    model_used: str


@functools.lru_cache(maxsize=1)
def _load_model():
    """Cached so we don't hit the Model Registry on every request."""
    mr = get_model_registry()
    hw_model = mr.get_model(config.MODEL_NAME)
    model_dir = hw_model.download()
    # NOTE: adjust filename to match whatever training_pipeline.py saved
    model = joblib.load(f"{model_dir}/{hw_model.name}.joblib")
    return model, hw_model.version


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict", response_model=ForecastResponse)
def predict():
    model, version = _load_model()

    # NOTE: Hopsworks' feature-view "online vector" lookup (fv.get_feature_vector)
    # is built for multi-entity real-time key lookups (e.g. "give me features for
    # user_id=123"). This project has a single entity (Lahore) with a constantly
    # advancing timestamp as primary key, so the simpler and more correct approach
    # is just reading the latest row straight from the feature group.
    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME,
                               version=config.FEATURE_GROUP_VERSION)
    recent_df = fg.read()  # for a small feature group this is fine; for scale,
    # filter server-side instead of reading everything, e.g. fg.filter(...)
    latest_row = recent_df.sort_values("timestamp").iloc[-1]

    feature_cols = [c for c in recent_df.columns
                    if c not in {"timestamp", "aqi_target_24h",
                                 "aqi_target_48h", "aqi_target_72h"}]
    latest_features = latest_row[feature_cols].values.reshape(1, -1)

    prediction = model.predict(latest_features)[0]

    hazardous = prediction >= HAZARDOUS_THRESHOLD

    return ForecastResponse(
        city=config.CITY_NAME,
        forecast_72h=[{"horizon_hours": 72, "predicted_aqi": float(prediction)}],
        hazardous_alert=bool(hazardous),
        model_used=f"v{version}",
    )


@app.get("/model-metrics")
def model_metrics():
    """Used by the Streamlit 'Model Performance' page."""
    mr = get_model_registry()
    hw_model = mr.get_model(config.MODEL_NAME)
    return hw_model.training_metrics
