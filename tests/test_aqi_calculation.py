"""EPA AQI tests."""
import numpy as np
from src.utils.aqi_calculation import pm25_to_aqi


def test_epa_worked_examples():
    """EPA example numbers."""
    assert pm25_to_aqi(35.5) == 101
    assert pm25_to_aqi(9.0) == 50
    assert pm25_to_aqi(0) == 0


def test_no_negative_aqi_across_full_range():
    """AQI must stay >= 0 from 0 to 400 µg/m³."""
    for i in range(4001):
        pm25 = i / 10
        aqi = pm25_to_aqi(pm25)
        assert aqi >= 0, f"pm25={pm25} produced negative AQI={aqi}"


def test_aqi_is_monotonic_non_decreasing():
    """Higher PM2.5 should not give a lower AQI."""
    prev_aqi = -1
    for i in range(4001):
        pm25 = i / 10
        aqi = pm25_to_aqi(pm25)
        assert aqi >= prev_aqi, f"AQI decreased at pm25={pm25}"
        prev_aqi = aqi


def test_specific_gap_value_that_previously_broke():
    """9.0989 used to fall in a gap and break."""
    result = pm25_to_aqi(9.0989)
    assert result >= 0
    assert 40 <= result <= 60  # should land near the 50 (Good/Moderate boundary)


def test_null_input_handled():
    assert pm25_to_aqi(np.nan) is None
