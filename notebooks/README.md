# Notebooks

- `01_eda.ipynb` — EDA on the **raw** merged pollution+weather snapshot
  (`data/raw/aqi_raw_merged.parquet`). Run after:
  `python -m src.feature_pipeline raw-snapshot`
  Covers univariate/bivariate/multivariate analysis, ACF/PACF, seasonal
  decomposition, ADF stationarity, and smog-vs-normal comparison.
  Ends with a **Findings for FE** section that must be reviewed before any
  feature-engineering changes.

- `02_training.ipynb` — Colab GPU training for the **0–500 regression** path
  (aligned with `src/training_pipeline.py`). Full training also lives in
  `src/training_pipeline.py` via `python -m src.training_pipeline`.
