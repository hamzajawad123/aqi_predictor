"""
Alerts page — hazardous AQI level warnings (required by the project brief).
"""
import os
import requests
import streamlit as st

st.set_page_config(page_title="Alerts · AQI Predictor", page_icon="🚨", layout="wide")
st.title("🚨 Hazardous AQI Alerts")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

AQI_LEVELS = {
    1: ("Good", "🟢"),
    2: ("Fair", "🟡"),
    3: ("Moderate", "🟠"),
    4: ("Poor", "🔴"),
    5: ("Very Poor", "🟣"),
}

try:
    resp = requests.get(f"{API_BASE_URL}/predict", timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data["hazardous_alert"]:
        st.error(
            "⚠️ **Hazardous AQI predicted in the next 3 days.** "
            "Sensitive groups should limit outdoor exposure."
        )
    else:
        st.success("✅ No hazardous AQI levels predicted in the next 3 days.")

    st.subheader("AQI Scale Reference")
    for level, (label, emoji) in AQI_LEVELS.items():
        st.write(f"{emoji} **{level} — {label}**")

except requests.exceptions.RequestException as e:
    st.warning(f"Could not reach the prediction API: {e}")
