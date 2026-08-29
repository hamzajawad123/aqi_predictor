"""Turn PM2.5 into a US EPA AQI number. OpenWeather's 1–5 score is not used as the target."""
import pandas as pd

# PM2.5 low, PM2.5 high, AQI low, AQI high
PM25_BREAKPOINTS_2024 = [
    (0.0, 9.0, 0, 50),        # Good
    (9.1, 35.4, 51, 100),     # Moderate
    (35.5, 55.4, 101, 150),   # Unhealthy for Sensitive Groups
    (55.5, 125.4, 151, 200),  # Unhealthy
    (125.5, 225.4, 201, 300), # Very Unhealthy
    (225.5, 325.4, 301, 500), # Hazardous
]


def pm25_to_aqi(pm25: float) -> float:
    """One PM2.5 reading (µg/m³) to EPA AQI. Cut to 1 decimal first, do not round."""
    if pd.isna(pm25):
        return None
    pm25 = max(pm25, 0.0)
    pm25 = int(pm25 * 10) / 10  # keep 1 decimal, like EPA

    for c_lo, c_hi, aqi_lo, aqi_hi in PM25_BREAKPOINTS_2024:
        if c_lo <= pm25 <= c_hi:
            return round((aqi_hi - aqi_lo) / (c_hi - c_lo) * (pm25 - c_lo) + aqi_lo)

    # Very high PM2.5: keep going past 500 instead of capping.
    c_lo, c_hi, aqi_lo, aqi_hi = PM25_BREAKPOINTS_2024[-1]
    return round((aqi_hi - aqi_lo) / (c_hi - c_lo) * (pm25 - c_lo) + aqi_lo)


def pm25_series_to_aqi(pm25_series: pd.Series) -> pd.Series:
    """Same as pm25_to_aqi, one row at a time."""
    return pm25_series.apply(pm25_to_aqi)
