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


def _feature_names(fg) -> set[str]:
    feats = getattr(fg, "features", None) or []
    return {str(f.name).lower() for f in feats}


def get_or_create_feature_group(fs, df_for_schema=None):
    """
    Open FEATURE_GROUP_NAME @ FEATURE_GROUP_VERSION.

    An existing group keeps its original schema — Hopsworks will not add
    columns on insert. Hourly CI used to point at a pre-delta version
    (via a GitHub secret), then fail when aqi_delta_{24,48,72}h were new.
    If this group already has columns but is missing those deltas, fail
    here with a clear version message instead of a schema traceback.
    A brand-new group has an empty feature list; the first insert sets it
    from the caller's frame (`df_for_schema` is that frame).
    """
    print(
        f"[hopsworks] Feature group {config.FEATURE_GROUP_NAME} "
        f"v{config.FEATURE_GROUP_VERSION}"
    )
    fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        description="Hourly AQI + weather features for Lahore",
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )
    names = _feature_names(fg)
    if names:
        required = [f"aqi_delta_{h}h" for h in config.TARGET_HORIZONS]
        missing = [c for c in required if c.lower() not in names]
        if missing:
            raise RuntimeError(
                f"{config.FEATURE_GROUP_NAME} v{config.FEATURE_GROUP_VERSION} "
                f"is missing {missing}. That version was created before delta "
                f"targets, and Hopsworks cannot add columns on insert. Set "
                f"FEATURE_GROUP_VERSION to the current group (default 4) or "
                f"a new unused version, then run "
                f"`python -m src.feature_pipeline push-features` so the "
                f"schema includes aqi_delta_{{24,48,72}}h. Leave older "
                f"versions as rollback."
            )
    return fg


def get_or_create_feature_view(fs, fg, label_col: str = "aqi_delta_72h"):
    """
    A Feature View is a saved query/schema on top of the feature group.
    Default label is the primary 72h delta target (post-EDA FE).
    """
    query = fg.select_all()
    fv = fs.get_or_create_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=1,
        description="AQI feature view for training + online inference (delta targets)",
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