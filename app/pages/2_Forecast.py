"""
Forecast page - 3-day AQI forecast (day 1 / 2 / 3) from FastAPI /predict.
"""
import os
import requests
import pandas as pd
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("AQI Forecast - Lahore")
st.caption("Day 1 (24h) | Day 2 (48h) | Day 3 (72h)")


def get_forecast():
    resp = requests.get(f"{API_BASE_URL}/predict", timeout=30)
    resp.raise_for_status()
    return resp.json()


try:
    data = get_forecast()
    if data.get("current_aqi") is not None:
        st.metric("Current AQI", f"{data['current_aqi']:.0f}")

    forecasts = data.get("forecasts") or data.get("forecast_72h") or []
    forecast_df = pd.DataFrame(forecasts)
    if "day" not in forecast_df.columns and "horizon_hours" in forecast_df.columns:
        forecast_df["day"] = (forecast_df["horizon_hours"] // 24).astype(int)

    st.subheader("3-day forecast")
    if "predicted_aqi" in forecast_df.columns:
        chart_df = forecast_df.dropna(subset=["predicted_aqi"])
        if not chart_df.empty:
            st.line_chart(chart_df.set_index("horizon_hours")["predicted_aqi"])
    st.dataframe(forecast_df, use_container_width=True)

    if data.get("hazardous_alert"):
        st.error("Hazardous AQI predicted within the next 3 days.")
    else:
        st.success("No hazardous AQI levels predicted in the next 3 days.")
except Exception as e:
    st.warning(f"Could not reach the prediction API: {e}")
