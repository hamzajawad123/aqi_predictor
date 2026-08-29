"""EPA AQI bands and colors."""
from __future__ import annotations

EPA_BANDS = (
    (0, 50, "Good", "#00E400"),
    (50, 100, "Moderate", "#FFD100"),
    (100, 150, "Unhealthy for sensitive groups", "#FF7E00"),
    (150, 200, "Unhealthy", "#FF0000"),
    (200, 300, "Very unhealthy", "#8F3F97"),
    (300, 500, "Hazardous", "#7E0023"),
)


def epa_category(aqi: float | None) -> str:
    if aqi is None:
        return "Unknown"
    for _lo, hi, label, _color in EPA_BANDS:
        if aqi <= hi:
            return label
    return "Hazardous"


def epa_color(aqi: float | None) -> str:
    if aqi is None:
        return "#8B9BB4"
    for _lo, hi, _label, color in EPA_BANDS:
        if aqi <= hi:
            return color
    return "#7E0023"
