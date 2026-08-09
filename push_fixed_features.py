"""
Run ONCE, manually, after fix_features_pipeline_v2.py has produced
features_fixed.csv. Loads the repaired historical data into a NEW version
of the aqi_features feature group (v2), instead of touching v1 in place.

Why a new version instead of overwriting v1: v1's schema was created before
aqi_target_72h existed, and its stored rows have the row-shifted (broken)
lag/rolling values. A clean v2 avoids any ambiguity about whether every row
in the store is trustworthy, and gives you an easy rollback (v1 still exists,
untouched, if anything looks wrong).

After this runs successfully:
  1. Set FEATURE_GROUP_VERSION=2 in your .env (or GitHub Actions secrets).
  2. Set FEATURE_VIEW_NAME=aqi_feature_view_v2 in your .env too (a feature
     view is bound to the fg version it was created against -- reusing the
     old name against a new fg version can error or silently point at stale
     metadata depending on SDK version, so a new name sidesteps that).
  3. Run: python -m src.utils.hopsworks_utils
     This creates the v2 feature view that api/main.py and training read from.
  4. Re-run src/training_pipeline.py (or the Colab notebook) against v2.

Place this file at the project root (same level as the `src/` folder) before
running, so `from src...` imports resolve.
"""
import pandas as pd
from src import config
from src.utils.hopsworks_utils import get_feature_store

FIXED_CSV_PATH = "features_fixed.csv"
NEW_VERSION = 2

df = pd.read_csv(FIXED_CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

if "openweather_aqi_category" in df.columns:
    df["openweather_aqi_category"] = df["openweather_aqi_category"].astype("int64")

print(f"Loaded {len(df)} rows x {len(df.columns)} cols from {FIXED_CSV_PATH}")

fs = get_feature_store()
fg = fs.get_or_create_feature_group(
    name=config.FEATURE_GROUP_NAME,
    version=NEW_VERSION,
    description="Hourly AQI + weather features for Lahore (v2 -- gaps filled, "
                 "72h target added, 4 contradictory rows dropped)",
    primary_key=["timestamp"],
    event_time="timestamp",
    time_travel_format="HUDI",
)

fg.insert(df)
print(f"Inserted {len(df)} rows into '{config.FEATURE_GROUP_NAME}' v{NEW_VERSION}.")
print("\nNext: set FEATURE_GROUP_VERSION=2 and FEATURE_VIEW_NAME=aqi_feature_view_v2 "
      "in your .env, then run: python -m src.utils.hopsworks_utils")
