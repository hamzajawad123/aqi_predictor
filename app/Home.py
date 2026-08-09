"""
Streamlit entrypoint. Run with: streamlit run app/Home.py
Additional pages live in app/pages/ and appear automatically in the sidebar
(Streamlit's `pages/` directory convention).
"""
import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌫️",
    layout="wide",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


@st.cache_data(ttl=3600)  # re-fetch at most once an hour, matches feature pipeline cadence
def get_forecast():
    resp = requests.get(f"{API_BASE_URL}/predict", timeout=10)
    resp.raise_for_status()
    return resp.json()


st.title("🌫️ Pearls AQI Predictor — Lahore")
st.caption("3-day Air Quality Index forecast · Serverless FTI pipeline · 10Pearls Internship")

try:
    data = get_forecast()
    col1, col2, col3 = st.columns(3)
    col1.metric("City", data["city"])
    col2.metric("Model in use", data["model_used"])
    col3.metric("Hazardous alert", "🔴 YES" if data["hazardous_alert"] else "🟢 No")

    if data["hazardous_alert"]:
        st.error("⚠️ Hazardous AQI levels predicted in the next 3 days. "
                 "Consider limiting outdoor exposure.")

    st.subheader("72-hour forecast")
    st.json(data["forecast_72h"])

except requests.exceptions.RequestException as e:
    st.warning(f"Could not reach the prediction API at {API_BASE_URL}. "
               f"Is the FastAPI service running? ({e})")

st.divider()
st.markdown(
    "Use the sidebar to explore **EDA**, the full **Forecast** view, "
    "**Alerts** history, and **Model Performance** (RMSE/MAE/R² + SHAP)."
)
