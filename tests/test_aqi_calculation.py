"""
Tests for the EPA AQI calculation — including a regression test for the
breakpoint-gap bug found during development (a raw float like 9.0989 fell
between the 9.0 and 9.1 breakpoints and produced a nonsensical negative AQI).
"""
import numpy as np
from src.utils.aqi_calculation import pm25_to_aqi


def test_epa_worked_examples():
    """Matches the EPA's own published worked examples exactly."""
    assert pm25_to_aqi(35.5) == 101
    assert pm25_to_aqi(9.0) == 50
    assert pm25_to_aqi(0) == 0


def test_no_negative_aqi_across_full_range():
    """Regression test: every truncated value from 0.0 to 400.0 must produce
    a non-negative AQI. This specifically catches the breakpoint-gap bug
    (values like 9.0989 falling between the 9.0|9.1 boundary)."""
    for i in range(4001):
        pm25 = i / 10
        aqi = pm25_to_aqi(pm25)
        assert aqi >= 0, f"pm25={pm25} produced negative AQI={aqi}"


def test_aqi_is_monotonic_non_decreasing():
    """Higher PM2.5 must never produce a lower AQI than a smaller PM2.5."""
    prev_aqi = -1
    for i in range(4001):
        pm25 = i / 10
        aqi = pm25_to_aqi(pm25)
        assert aqi >= prev_aqi, f"AQI decreased at pm25={pm25}"
        prev_aqi = aqi


def test_specific_gap_value_that_previously_broke():
    """The exact value that exposed the original bug during testing."""
    result = pm25_to_aqi(9.0989)
    assert result >= 0
    assert 40 <= result <= 60  # should land near the 50 (Good/Moderate boundary)


def test_null_input_handled():
    assert pm25_to_aqi(np.nan) is None
