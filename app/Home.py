"""AQI dashboard. Run: streamlit run app/Home.py"""
from __future__ import annotations

import html as html_lib

import streamlit as st

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from bootstrap import setup

setup()

from charts import (  # noqa: E402
    CHART_CFG,
    POLLUTANT_COLORS,
    correlation_heatmap,
    drivers_bar,
    forecast_trend,
    gauge,
    history_line,
    monthly_box,
    pm25_hist,
    pollutant_bars,
    trend_24h,
)
from client import get_dashboard  # noqa: E402
from theme import EPA_BANDS, epa_category, epa_color  # noqa: E402

st.markdown(
    """
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap");
    html, body, .stApp { font-family: "Source Sans 3", "Segoe UI", sans-serif; }
    [data-testid="stSidebar"], [data-testid="collapsedControl"], footer {
      display: none !important;
    }
    .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1140px; }
    div[data-testid="stPlotlyChart"] { overflow: hidden; }

    .hero-kicker {
      font-size: 0.72rem; font-weight: 700; letter-spacing: 0.16em;
      text-transform: uppercase; color: var(--text-color); opacity: 0.55; margin-bottom: 0.2rem;
    }
    .hero-title {
      font-size: 1.9rem; font-weight: 700; color: var(--text-color);
      letter-spacing: -0.03em; margin: 0;
    }
    .hero-sub { color: var(--text-color); opacity: 0.72; font-size: 0.95rem; margin: 0.35rem 0 0.7rem; }
    .pill {
      display: inline-block; font-size: 0.8rem; font-weight: 700;
      padding: 0.32rem 0.8rem; border-radius: 999px;
    }
    .pill-ok { background: color-mix(in srgb, #2A9D8F 22%, transparent); color: #127A6A; }
    .pill-alert { background: color-mix(in srgb, #E76F51 22%, transparent); color: #B23A22; }
    [data-testid="stAppViewContainer"] .pill-ok { color: var(--text-color); }
    [data-testid="stAppViewContainer"] .pill-alert { color: var(--text-color); }

    .legend {
      display: flex; flex-wrap: wrap; gap: 0.4rem 0.85rem;
      margin: 0.15rem 0 0.15rem; padding: 0.35rem 0.1rem 0.1rem;
    }
    .legend span {
      display: inline-flex; align-items: center; gap: 0.35rem;
      font-size: 0.74rem; font-weight: 600;
    }
    .swatch { width: 10px; height: 10px; border-radius: 999px; display: inline-block; flex-shrink: 0; }

    .loader-wrap {
      margin: 1.4rem 0 2.2rem; padding: 2.1rem 1.4rem 1.9rem;
      border-radius: 18px; text-align: center;
      background: #1A2740; color: #F4F7FB;
      border: 1px solid #2E3D59;
    }
    .spinner {
      width: 42px; height: 42px; margin: 0 auto 1rem;
      border-radius: 50%;
      border: 3px solid #2E3D59;
      border-top-color: #2A9D8F;
      animation: spin 0.8s linear infinite;
    }
    .loader-title { font-size: 1.05rem; font-weight: 700; }
    .loader-sub { font-size: 0.88rem; opacity: 0.65; margin-top: 0.3rem; }
    @keyframes spin { to { transform: rotate(360deg); } }
    </style>
    """,
    unsafe_allow_html=True,
)

POLLUTANT_META = (
    ("pm2_5", "PM2.5", "µg/m³"),
    ("pm10", "PM10", "µg/m³"),
    ("o3", "O₃", "µg/m³"),
    ("no2", "NO₂", "µg/m³"),
    ("so2", "SO₂", "µg/m³"),
    ("co", "CO", "µg/m³"),
)
WEATHER_META = (
    ("temperature", "Temperature", "°C", "#E76F51"),
    ("humidity", "Humidity", "%", "#4EA8DE"),
    ("pressure", "Pressure", "hPa", "#6C7A89"),
    ("wind_speed", "Wind speed", "m/s", "#2A9D8F"),
)


def show_chart(fig, key: str) -> None:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=CHART_CFG,
        key=key,
        theme="streamlit",
    )


def section(title: str, caption: str | None = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


def _is_light() -> bool:
    try:
        return str(st.get_option("theme.base") or "").lower() == "light"
    except Exception:
        return False


def _palette() -> dict[str, str]:
    if _is_light():
        return {
            "card_bg": "#FFFFFF",
            "card_border": "#D7E0EC",
            "text": "#122033",
            "muted": "#5B6B7C",
            "shadow": "0 8px 22px rgba(18,32,51,0.08)",
        }
    return {
        "card_bg": "#1A2740",
        "card_border": "#3A4D6A",
        "text": "#F4F7FB",
        "muted": "#9AABC0",
        "shadow": "0 10px 28px rgba(0,0,0,0.38)",
    }


def legend_html() -> str:
    pal = _palette()
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'font-size:12px;font-weight:600;color:{pal["text"]}">'
        f'<i class="swatch" style="background:{c}"></i>{html_lib.escape(lab)}</span>'
        for _a, _b, lab, c in EPA_BANDS
    )
    return f'<div class="legend">{chips}</div>'


def card(label: str, value: str, unit: str, accent: str, extra: str = "", value_size: str = "26px") -> str:
    pal = _palette()
    return (
        f'<div style="'
        f"background:{pal['card_bg']};"
        f"border:1px solid {pal['card_border']};"
        f"border-left:5px solid {accent};"
        f"border-radius:16px;"
        f"padding:16px 16px 14px 16px;"
        f"box-shadow:{pal['shadow']};"
        f'">'
        f'<div style="font-size:11px;font-weight:700;letter-spacing:0.08em;'
        f"text-transform:uppercase;color:{pal['muted']};\">{html_lib.escape(label)}</div>"
        f"{extra}"
        f'<div style="font-size:{value_size};font-weight:700;letter-spacing:-0.03em;'
        f"color:{accent};margin-top:6px;\">{html_lib.escape(value)}</div>"
        f'<div style="font-size:12px;font-weight:600;color:{pal["muted"]};margin-top:4px;">'
        f"{html_lib.escape(unit)}</div>"
        f"</div>"
    )


def card_grid(cards: list[str], columns: int) -> None:
    min_w = {6: "140px", 4: "160px", 3: "210px"}.get(columns, "160px")
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax({min_w},1fr));'
        f'gap:12px;margin:4px 0 8px;">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero-kicker">10Pearls Internship · Serverless FTI</div>
    <h1 class="hero-title">Pearls AQI Predictor</h1>
    <p class="hero-sub">3-day air quality forecast for Lahore from live pollutants, weather, and seasonal patterns.</p>
    """,
    unsafe_allow_html=True,
)

loader = st.empty()
with loader:
    st.markdown(
        """
        <div class="loader-wrap">
          <div class="spinner"></div>
          <div class="loader-title">Loading Lahore forecast</div>
          <div class="loader-sub">Fetching the latest features and running the 24 / 48 / 72 hour models…</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

try:
    data = get_dashboard()
except Exception as e:
    loader.empty()
    st.warning(
        "Could not load the forecast from Hopsworks. "
        "Locally, keep `.env` filled. On Streamlit Cloud, add the same keys as secrets. "
        f"({e})"
    )
    st.stop()

loader.empty()

city = data["city"]
aqi = data["current_aqi"]
cat = epa_category(aqi)
color = epa_color(aqi)
updated = data["timestamp"].strftime("%b %d, %Y · %I:%M %p PKT")
poll = data.get("pollutants") or {}
wx = data.get("weather") or {}
forecasts = data.get("forecasts") or []
pill_cls = "pill-alert" if data["hazardous_alert"] else "pill-ok"
pill_txt = "Hazardous alert" if data["hazardous_alert"] else "No hazardous alert"

st.markdown(f'<span class="pill {pill_cls}">{pill_txt}</span>', unsafe_allow_html=True)

st.divider()
section(f"{city} air quality", f"US EPA AQI from PM2.5 · updated {updated}")

gcol, scol = st.columns([0.46, 0.54], gap="large")
with gcol:
    show_chart(gauge(aqi, data.get("aqi_24h_ago")), "chart_gauge")
    st.markdown(legend_html(), unsafe_allow_html=True)
with scol:
    pal = _palette()
    st.markdown(
        f'<div style="background:{pal["card_bg"]};border:1px solid {pal["card_border"]};'
        f'border-left:5px solid {color};border-radius:16px;padding:18px 18px 16px;'
        f'box-shadow:{pal["shadow"]};">'
        f'<div style="font-size:11px;font-weight:700;letter-spacing:0.08em;'
        f'text-transform:uppercase;color:{pal["muted"]};">Current air quality</div>'
        f'<div style="font-size:22px;font-weight:700;color:{color};margin-top:8px;">'
        f'{html_lib.escape(cat)}</div>'
        f'<div style="font-size:15px;font-weight:700;color:{pal["text"]};margin-top:6px;">AQI {aqi:.0f}</div>'
        f'<div style="font-size:12px;font-weight:600;color:{pal["muted"]};margin-top:6px;">'
        f'Change vs 24 hours ago is shown on the gauge</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()
section("Current pollutants", "Latest hourly concentrations")
poll_cards = []
for key, label, unit in POLLUTANT_META:
    val = poll.get(key)
    poll_cards.append(
        card(
            label,
            f"{val:.1f}" if val is not None else "—",
            unit,
            POLLUTANT_COLORS.get(label, "#4ECDC4"),
        )
    )
card_grid(poll_cards, 6)

st.divider()
section("24-hour AQI trend", "Hourly US EPA AQI · last 24 hours (Pakistan time)")
hist = data.get("history_24h")
if hist is not None and len(hist) > 1:
    with st.container(border=True):
        show_chart(trend_24h(hist), "chart_trend_24h")

st.divider()
section("Current weather", "Open-Meteo conditions at the latest hour")
wx_cards = []
for key, label, unit, accent in WEATHER_META:
    val = wx.get(key)
    if val is None:
        shown = "—"
    elif key == "humidity":
        shown = f"{val:.0f}"
    else:
        shown = f"{val:.1f}"
    wx_cards.append(card(label, shown, unit, accent))
card_grid(wx_cards, 4)

st.divider()
section("3-day forecast", "Predicted US EPA AQI at 24, 48 and 72 hours")
fc_cards = []
for item in forecasts:
    pred = item.get("predicted_aqi")
    day = item.get("day")
    if pred is None:
        fc_cards.append(card(f"+{day} day", "—", "Unavailable", "#6C7A89", value_size="34px"))
        continue
    pcat = epa_category(pred)
    pcol = epa_color(pred)
    rmse = item.get("rmse")
    unit = f"Model RMSE ± {rmse:.1f}" if rmse else "US EPA AQI"
    extra = (
        f'<div style="font-size:14px;font-weight:700;color:{pcol};margin-top:8px;line-height:1.3;">'
        f"{html_lib.escape(pcat)}</div>"
    )
    fc_cards.append(card(f"+{day} day", f"{pred:.0f}", unit, pcol, extra=extra, value_size="34px"))
card_grid(fc_cards, 3)

st.divider()
section("Predicted AQI over the next 3 days", "Today plus the three model horizons")
with st.container(border=True):
    show_chart(forecast_trend(aqi, forecasts), "chart_forecast")

if poll:
    st.divider()
    section("Pollutant snapshot", "Same latest hour as the cards above, as a bar chart")
    with st.container(border=True):
        show_chart(pollutant_bars(poll), "chart_pollutants")

drivers = data.get("drivers") or []
if drivers:
    st.divider()
    section(
        "Why this prediction",
        "How current conditions compare with the last 30 days (not model SHAP weights)",
    )
    with st.container(border=True):
        show_chart(drivers_bar(drivers), "chart_drivers")

st.divider()
section("Historical AQI", "Daily mean US EPA AQI from the feature store")
daily = data.get("history_daily")
if daily is not None and len(daily) > 2:
    with st.container(border=True):
        show_chart(history_line(daily), "chart_history")

recent = data.get("recent")
if recent is not None and len(recent) > 20:
    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        section("PM2.5 distribution", "Hourly PM2.5 over the last 90 days")
        with st.container(border=True):
            show_chart(pm25_hist(recent), "chart_pm25")
    with right:
        section("AQI spread by month", "Hourly AQI grouped by month (last 90 days)")
        with st.container(border=True):
            show_chart(monthly_box(recent), "chart_box")

    st.divider()
    section("Pollutant correlation", "Pearson correlation on the last 90 days of hourly data")
    with st.container(border=True):
        show_chart(correlation_heatmap(recent), "chart_corr")
