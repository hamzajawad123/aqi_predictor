"""
US EPA Air Quality Index (AQI) calculation from PM2.5 concentration.

WHY THIS EXISTS: OpenWeather's own `main.aqi` field is NOT a continuous AQI —
it's a coarse 1-5 categorical index (1=Good ... 5=Very Poor), documented at
https://openweathermap.org/api/air-pollution. Training regression models
(RMSE/MAE/R²) against a 5-value categorical field doesn't match what "AQI"
means in the project brief — a continuous number, like the "82" shown in the
brief's own AQICN screenshot. So `aqi` throughout this project is computed
here, from the raw PM2.5 concentration, using the real US EPA formula —
OpenWeather's original 1-5 field is kept as a separate reference column
(`openweather_aqi_category`), never used as the model target.

BREAKPOINTS: US EPA's 2024-revised PM2.5 table (effective 6 May 2024) —
verified against the EPA's Federal Register final rule before implementing:
"the EPA is revising the AQI value of 50 to 9.0 µg/m3 and is retaining the
AQI values of 100 and 150 at 35.4 µg/m3 and 55.4 µg/m3... revising the AQI
values of 200, 300 and 500 to 125.4 µg/m3, 225.4 µg/m3, and 325.4 µg/m3."
(Federal Register, Reconsideration of the NAAQS for Particulate Matter, 2024)

SIMPLIFICATION (documented, not hidden): the real EPA AQI takes the MAX
across all 6 criteria pollutants (PM2.5, PM10, O3, NO2, SO2, CO), each with
its own breakpoint table. This implementation uses PM2.5 only, since PM2.5
is well-documented as the dominant pollutant in the large majority of
readings — and specifically for South Asian smog/particulate pollution like
Lahore's. Extending to a true multi-pollutant max would need verified
breakpoint tables for the other 5 pollutants too.
"""
import pandas as pd

# (concentration_low, concentration_high, aqi_low, aqi_high) — µg/m³
PM25_BREAKPOINTS_2024 = [
    (0.0, 9.0, 0, 50),        # Good
    (9.1, 35.4, 51, 100),     # Moderate
    (35.5, 55.4, 101, 150),   # Unhealthy for Sensitive Groups
    (55.5, 125.4, 151, 200),  # Unhealthy
    (125.5, 225.4, 201, 300), # Very Unhealthy
    (225.5, 325.4, 301, 500), # Hazardous
]


def pm25_to_aqi(pm25: float) -> float:
    """Convert a single PM2.5 concentration (µg/m³) to US EPA AQI.

    IMPORTANT: the EPA breakpoint table has small gaps between adjacent bands
    (9.0 | 9.1, 35.4 | 35.5, etc.) — this is intentional in the official
    methodology, which TRUNCATES the concentration to 1 decimal place before
    the lookup, so a value can only ever land exactly on a documented
    breakpoint, never in a gap. Skipping that truncation step is a real bug:
    a raw float like 9.0989 (perfectly plausible sensor data) falls between
    9.0 and 9.1, matches no band, and falls through to the wrong branch —
    this was caught during testing and is fixed here by truncating first.
    """
    if pd.isna(pm25):
        return None
    pm25 = max(pm25, 0.0)
    pm25 = int(pm25 * 10) / 10  # truncate (not round) to 1 decimal, per EPA method

    for c_lo, c_hi, aqi_lo, aqi_hi in PM25_BREAKPOINTS_2024:
        if c_lo <= pm25 <= c_hi:
            return round((aqi_hi - aqi_lo) / (c_hi - c_lo) * (pm25 - c_lo) + aqi_lo)

    # Above the top breakpoint (325.4 µg/m³) — extrapolate from the last band
    # rather than hard-capping at 500, so extreme smog-season readings
    # (realistic for Lahore) still produce a meaningful, ranked value instead
    # of every severe day collapsing to the same "500".
    c_lo, c_hi, aqi_lo, aqi_hi = PM25_BREAKPOINTS_2024[-1]
    return round((aqi_hi - aqi_lo) / (c_hi - c_lo) * (pm25 - c_lo) + aqi_lo)


def pm25_series_to_aqi(pm25_series: pd.Series) -> pd.Series:
    """Vectorized (row-wise) version for a pandas Series of PM2.5 readings."""
    return pm25_series.apply(pm25_to_aqi)
