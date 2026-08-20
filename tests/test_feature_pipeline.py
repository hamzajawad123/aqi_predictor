"""Hourly-pipeline helpers that don't need a live Hopsworks connection."""
import pandas as pd

from src.feature_pipeline import _as_utc, _unix_utc


def test_as_utc_treats_naive_and_offset_strings_as_the_same_hour():
    naive = pd.Series([pd.Timestamp("2026-08-18 05:00:00")])
    hopsworks_style = pd.Series(["2026-08-18 05:00:00+00:00"])

    existing = set(_as_utc(hopsworks_style))
    is_new = ~_as_utc(naive).isin(existing)

    assert not bool(is_new.iloc[0])


def test_as_utc_string_compare_would_have_missed():
    naive = pd.Series([pd.Timestamp("2026-08-18 05:00:00")])
    hopsworks_style = pd.Series(["2026-08-18 05:00:00+00:00"])

    # The old .astype(str) path treats these as different keys.
    assert naive.astype(str).iloc[0] != hopsworks_style.astype(str).iloc[0]


def test_unix_utc_treats_naive_hours_as_utc():
    naive = pd.Timestamp("2026-08-18 05:00:00")
    aware = pd.Timestamp("2026-08-18 05:00:00", tz="UTC")
    assert _unix_utc(naive) == _unix_utc(aware) == int(aware.timestamp())
