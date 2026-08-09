"""
EDA page — reads the local snapshot exported by notebooks/01_eda.ipynb
(data/eda_snapshot.parquet) rather than querying Hopsworks on every page
load, so the dashboard stays fast.
"""
import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EDA · AQI Predictor", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

SNAPSHOT_PATH = "data/eda_snapshot.parquet"

if not os.path.exists(SNAPSHOT_PATH):
    st.warning(
        f"No EDA snapshot found at `{SNAPSHOT_PATH}`. Run "
        "`notebooks/01_eda.ipynb` once (it saves this file automatically) "
        "after you've backfilled historical data."
    )
    st.stop()

df = pd.read_parquet(SNAPSHOT_PATH)

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
        st.info("wind_speed not in snapshot yet — available once the hourly "
                 "feature pipeline has been running (weather isn't in the "
                 "free historical backfill).")

st.subheader("Correlation heatmap")
numeric_df = df.select_dtypes(include="number")
st.plotly_chart(px.imshow(numeric_df.corr(), color_continuous_scale="RdBu_r",
                           zmin=-1, zmax=1), use_container_width=True)

st.caption(
    "Full time-series analysis (ACF/PACF, seasonal decomposition, ADF "
    "stationarity test) is in notebooks/01_eda.ipynb — see reports/final_report.md "
    "for the write-up."
)
