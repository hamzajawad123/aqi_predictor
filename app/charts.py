"""Plotly charts with a fixed height."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from theme import EPA_BANDS, epa_color

FONT = "Source Sans 3, Segoe UI, sans-serif"
PLOT = "#152033"
PAPER = "rgba(0,0,0,0)"
TEXT = "#E8EEF4"
MUTED = "#A9B6C7"
TITLE = "#F4F7FB"
GRID = "rgba(255,255,255,0.07)"
ALERT = 151
CHART_CFG = {"displayModeBar": False, "responsive": True}

POLLUTANT_COLORS = {
    "PM2.5": "#F4A261",
    "PM10": "#E76F51",
    "O₃": "#2A9D8F",
    "NO₂": "#4EA8DE",
    "SO₂": "#E9C46A",
    "CO": "#9B5DE5",
}


def _is_light() -> bool:
    try:
        import streamlit as st

        return str(st.get_option("theme.base") or "").lower() == "light"
    except Exception:
        return False


def style(
    fig: go.Figure,
    x_title: str,
    y_title: str,
    *,
    height: int = 380,
    y_range=None,
) -> go.Figure:
    light = _is_light()
    plot = "#FFFFFF" if light else "#152033"
    text = "#1B2430" if light else "#E8EEF4"
    muted = "#5A6A7A" if light else "#A9B6C7"
    title = "#122033" if light else "#F4F7FB"
    grid = "rgba(15,23,42,0.08)" if light else "rgba(255,255,255,0.07)"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=plot,
        font=dict(family=FONT, size=13, color=text),
        height=height,
        autosize=True,
        margin=dict(l=68, r=24, t=16, b=68),
        hoverlabel=dict(font_size=13, font_family=FONT),
        showlegend=False,
        bargap=0.18,
    )
    fig.update_xaxes(
        title=dict(text=f"<b>{x_title}</b>", font=dict(size=14, color=title, family=FONT)),
        tickfont=dict(size=12, color=muted, family=FONT),
        gridcolor=grid,
        zeroline=False,
        showline=True,
        linecolor="rgba(128,128,128,0.28)",
        automargin=True,
    )
    fig.update_yaxes(
        title=dict(text=f"<b>{y_title}</b>", font=dict(size=14, color=title, family=FONT)),
        tickfont=dict(size=12, color=muted, family=FONT),
        gridcolor=grid,
        zeroline=False,
        showline=True,
        linecolor="rgba(128,128,128,0.28)",
        range=y_range,
        automargin=True,
    )
    return fig


def gauge(aqi: float, delta_ref: float | None) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta" if delta_ref is not None else "gauge+number",
            value=aqi,
            number={"font": {"size": 40, "color": "#122033" if _is_light() else "#F4F7FB", "family": FONT}, "valueformat": ".0f"},
            delta=(
                {
                    "reference": delta_ref,
                    "valueformat": ".0f",
                    "increasing": {"color": "#FF6B6B"},
                    "decreasing": {"color": "#7CFF7C"},
                }
                if delta_ref is not None
                else None
            ),
            gauge={
                "axis": {"range": [0, 300], "tickwidth": 1, "tickcolor": MUTED},
                "bar": {"color": epa_color(aqi), "thickness": 0.28},
                "bgcolor": "#FFFFFF" if _is_light() else "#152033",
                "borderwidth": 0,
                "steps": [
                    {"range": [lo, hi], "color": color} for lo, hi, _label, color in EPA_BANDS
                ],
                "threshold": {
                    "line": {"color": "#FF6B6B", "width": 2},
                    "thickness": 0.75,
                    "value": ALERT,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color="#122033" if _is_light() else "#E8EEF4"),
        height=280,
        margin=dict(l=16, r=16, t=12, b=8),
        autosize=True,
    )
    return fig


def trend_24h(hist: pd.DataFrame) -> go.Figure:
    ts = pd.to_datetime(hist["timestamp"], utc=True).dt.tz_convert("Asia/Karachi")
    y_max = max(180.0, float(hist["aqi"].max()) + 25)
    fig = go.Figure()
    for lo, hi, _label, color in EPA_BANDS:
        if lo >= y_max:
            continue
        fig.add_hrect(y0=lo, y1=min(hi, y_max), fillcolor=color, opacity=0.09, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=hist["aqi"],
            mode="lines",
            line=dict(color="#5EEAD4", width=2.6),
            hovertemplate="%{x|%d %b %H:%M}<br><b>AQI %{y:.0f}</b><extra></extra>",
        )
    )
    return style(fig, "Time (PKT)", "US EPA AQI", height=380, y_range=[0, y_max])


def forecast_trend(current: float, forecasts: list[dict]) -> go.Figure:
    xs = ["Today"]
    ys = [current]
    for item in forecasts:
        if item.get("predicted_aqi") is None:
            continue
        xs.append(f"+{item['day']} day")
        ys.append(float(item["predicted_aqi"]))
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line=dict(color="#F4A261", width=3),
            marker=dict(
                size=14,
                color=[epa_color(y) for y in ys],
                line=dict(width=2, color="#0B1220"),
            ),
            hovertemplate="%{x}<br><b>AQI %{y:.0f}</b><extra></extra>",
        )
    )
    y_max = max(180.0, max(ys) + 25)
    fig.add_hline(y=ALERT, line_dash="dash", line_color="#FF6B6B", line_width=1.5)
    return style(fig, "Forecast horizon", "US EPA AQI", height=380, y_range=[0, y_max])


def pollutant_bars(pollutants: dict[str, float]) -> go.Figure:
    labels = {
        "pm2_5": "PM2.5",
        "pm10": "PM10",
        "o3": "O₃",
        "no2": "NO₂",
        "so2": "SO₂",
        "co": "CO",
    }
    xs = [labels.get(k, k) for k in pollutants]
    ys = list(pollutants.values())
    colors = [POLLUTANT_COLORS.get(x, "#5EEAD4") for x in xs]
    fig = go.Figure(
        go.Bar(
            x=xs,
            y=ys,
            marker_color=colors,
            hovertemplate="%{x}<br><b>%{y:.1f} µg/m³</b><extra></extra>",
        )
    )
    return style(fig, "Pollutant", "Concentration (µg/m³)", height=360)


def drivers_bar(drivers: list[dict]) -> go.Figure:
    top = list(reversed(drivers[:8]))
    fig = go.Figure(
        go.Bar(
            x=[d["delta_pct"] for d in top],
            y=[d["feature"] for d in top],
            orientation="h",
            marker_color=["#E76F51" if d["delta_pct"] >= 0 else "#2A9D8F" for d in top],
            hovertemplate="%{y}<br><b>%{x:.1f}% vs 30-day median</b><extra></extra>",
        )
    )
    return style(fig, "Change vs 30-day median (%)", "Condition", height=400)


def history_line(daily: pd.DataFrame) -> go.Figure:
    ts = pd.to_datetime(daily["timestamp"], utc=True).dt.tz_convert("Asia/Karachi")
    fig = go.Figure(
        go.Scatter(
            x=ts,
            y=daily["aqi"],
            mode="lines",
            line=dict(color="#4EA8DE", width=1.8),
            hovertemplate="%{x|%d %b %Y}<br><b>AQI %{y:.0f}</b><extra></extra>",
        )
    )
    return style(fig, "Date (PKT)", "Daily mean US EPA AQI", height=380)


def pm25_hist(recent: pd.DataFrame) -> go.Figure:
    if "pm2_5" not in recent.columns:
        return go.Figure()
    fig = go.Figure(
        go.Histogram(
            x=recent["pm2_5"],
            nbinsx=36,
            marker_color="#E9C46A",
            hovertemplate="PM2.5 %{x:.1f}<br>Hours %{y}<extra></extra>",
        )
    )
    return style(fig, "PM2.5 (µg/m³)", "Number of hours", height=380)


def monthly_box(recent: pd.DataFrame) -> go.Figure:
    frame = recent.copy()
    frame["month"] = (
        pd.to_datetime(frame["timestamp"], utc=True)
        .dt.tz_convert("Asia/Karachi")
        .dt.strftime("%b")
    )
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    present = [m for m in order if m in set(frame["month"])]
    fig = go.Figure()
    for month in present:
        fig.add_trace(
            go.Box(
                y=frame.loc[frame["month"] == month, "aqi"],
                name=month,
                marker_color="#9B5DE5",
                line=dict(color="#C77DFF"),
                fillcolor="rgba(155,93,229,0.35)",
                hovertemplate="%{x}<br>AQI %{y:.0f}<extra></extra>",
            )
        )
    fig.update_layout(boxmode="group")
    return style(fig, "Month (PKT)", "US EPA AQI", height=380)


def correlation_heatmap(recent: pd.DataFrame) -> go.Figure:
    cols = [
        c
        for c in (
            "aqi",
            "pm2_5",
            "pm10",
            "no2",
            "o3",
            "so2",
            "co",
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
        )
        if c in recent.columns
    ]
    corr = recent[cols].corr()
    labels = ["AQI" if c == "aqi" else c.replace("_", " ").title() for c in cols]
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title=dict(text="<b>r</b>")),
            hovertemplate="%{y} vs %{x}<br><b>%{z:.2f}</b><extra></extra>",
        )
    )
    fig = style(fig, "Variable", "Variable", height=520)
    fig.update_xaxes(tickangle=40, automargin=True)
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig
