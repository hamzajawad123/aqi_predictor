"""Dashboard data. Streamlit Cloud talks to Hopsworks via src.utils.serving."""
from __future__ import annotations

import streamlit as st

from src.utils.serving import dashboard_state
from theme import EPA_BANDS, epa_category, epa_color

__all__ = ["EPA_BANDS", "epa_category", "epa_color", "get_dashboard"]


@st.cache_data(ttl=300, show_spinner=False)
def get_dashboard() -> dict:
    return dashboard_state()
