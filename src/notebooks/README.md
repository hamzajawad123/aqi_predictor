# Notebooks

- `01_eda.ipynb` — the only required notebook (the brief asks for EDA).
  Univariate/bivariate/multivariate analysis, ACF/PACF, seasonal
  decomposition, ADF stationarity test, and a smog-vs-normal season
  comparison. Also exports `data/eda_snapshot.parquet`, which the Streamlit
  EDA page reads — run this at least once after backfilling data.

All actual model training (all 7 models: Persistence, Prophet, Ridge,
Random Forest, XGBoost, LightGBM, LSTM, GRU) lives in `.py` files under
`src/`, not notebooks — that's what the automated daily GitHub Action
actually runs. See `src/training_pipeline.py`.
