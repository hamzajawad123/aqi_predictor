"""
Forecast page — the core deliverable: 3-day AQI forecast with a chart,
pulled from the FastAPI /predict endpoint.
"""
import os
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Forecast · AQI Predictor", page_icon="🔮", layout="wide")
st.title("🔮 3-Day AQI Forecast")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


@st.cache_data(ttl=3600)
def get_forecast():
    resp = requests.get(f"{API_BASE_URL}/predict", timeout=10)
    resp.raise_for_status()
    return resp.json()


try:
    data = get_forecast()
    forecast_df = pd.DataFrame(data["forecast_72h"])
    st.line_chart(forecast_df.set_index("horizon_hours")["predicted_aqi"])
    st.dataframe(forecast_df, use_container_width=True)
except requests.exceptions.RequestException as e:
    st.warning(f"Could not reach the prediction API: {e}")
