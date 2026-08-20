"""
EDA page — reads the local raw merged snapshot written by
`python -m src.feature_pipeline raw-snapshot` (and upserted hourly)
rather than querying Hopsworks on every page load, so the dashboard stays fast.
"""
import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EDA · AQI Predictor", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

SNAPSHOT_PATHS = (
    "data/raw/aqi_raw_merged.parquet",
    "data/eda_snapshot.parquet",  # legacy export from older notebook runs
)

snapshot_path = next((p for p in SNAPSHOT_PATHS if os.path.exists(p)), None)

if snapshot_path is None:
    st.warning(
        "No raw AQI snapshot found. Run `python -m src.feature_pipeline raw-snapshot` "
        "to write `data/raw/aqi_raw_merged.parquet`."
    )
    st.stop()

df = pd.read_parquet(snapshot_path)

st.subheader("AQI over time")
st.plotly_chart(px.line(df, x="timestamp", y="aqi", title="AQI — Lahore"),
                 use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("AQI distribution")
    st.plotly_chart(px.histogram(df, x="aqi", nbins=40), use_container_width=True)

with col2:
    st.subheader("AQI vs. wind speed")
    if "wind_speed" in df.columns:
        st.plotly_chart(px.scatter(df, x="wind_speed", y="aqi", opacity=0.4),
                         use_container_width=True)
    else:
        st.info("wind_speed is missing from this snapshot — re-run `raw-snapshot`.")

st.subheader("Correlation heatmap")
numeric_df = df.select_dtypes(include="number")
st.plotly_chart(px.imshow(numeric_df.corr(), color_continuous_scale="RdBu_r",
                           zmin=-1, zmax=1), use_container_width=True)

st.caption(
    "Full time-series analysis (ACF/PACF, seasonal decomposition, ADF "
    "stationarity test) is in notebooks/01_eda.ipynb — see reports/final_report.md "
    "for the write-up."
)
