# Notebooks

- `01_eda.ipynb` — EDA on the **raw** merged pollution+weather snapshot
  (`data/raw/aqi_raw_merged.parquet`). Run after:
  `python -m src.feature_pipeline raw-snapshot`
  Covers univariate/bivariate/multivariate analysis, ACF/PACF, seasonal
  decomposition, ADF stationarity, and smog-vs-normal comparison.
  Ends with a **Findings for FE** section that must be reviewed before any
  feature-engineering changes.

- `colab_training.ipynb` — optional Colab GPU training. Prefer calling
  `from src.training_pipeline import train_and_evaluate` (or per-horizon
  helpers) so Colab stays aligned with CI. Full training lives in
  `src/training_pipeline.py` via `python -m src.training_pipeline`.
