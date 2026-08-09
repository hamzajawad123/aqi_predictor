"""
Thin wrapper around the Hopsworks SDK so feature_pipeline.py, training_pipeline.py,
and api/main.py all connect and read/write the same way.
"""
import os
import hopsworks
from src import config


def get_project():

    os.makedirs("/tmp", exist_ok=True)

    return hopsworks.login(
        host=config.HOPSWORKS_HOST,
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )


def get_feature_store():
    project = get_project()
    return project.get_feature_store()


def get_or_create_feature_group(fs, df_for_schema=None):
    fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        description="Hourly AQI + weather features for Lahore",
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI", 
    )
    return fg


def get_or_create_feature_view(fs, fg, label_col: str = "aqi_target_72h"):
    """
    A Feature View is what api/main.py's get_feature_vector() call needs to
    exist first — it's a saved query/schema on top of the feature group that
    enables fast online serving. Create this ONCE (e.g. run this file directly,
    or call it from a one-off setup script) before your FastAPI service can
    serve predictions.
    """
    query = fg.select_all()
    fv = fs.get_or_create_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=1,
        description="AQI feature view for training + online inference",
        labels=[label_col],
        query=query,
    )
    return fv


def get_model_registry():
    project = get_project()
    return project.get_model_registry()


if __name__ == "__main__":
    # One-time setup: run `python -m src.utils.hopsworks_utils` after your
    # first backfill to create the feature view the API depends on.
    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME,
                               version=config.FEATURE_GROUP_VERSION)
    fv = get_or_create_feature_view(fs, fg)
    print(f"[hopsworks_utils] Feature view '{fv.name}' v{fv.version} ready.")