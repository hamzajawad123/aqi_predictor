"""Save EDA figures from the local raw snapshot using the same plot recipes as notebooks/01_eda.ipynb."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from src import config
from src.utils.raw_io import load_raw_snapshot

OUT = Path(__file__).resolve().parent / "_report_figures"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

df = load_raw_snapshot()
df["month"] = df["timestamp"].dt.month
df["is_smog_season"] = df["month"].isin(config.SMOG_SEASON_MONTHS).astype(int)
pollutants = [c for c in ["pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3"] if c in df.columns]
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["month"], errors="ignore")

stats = []
stats.append(f"shape={df.shape}")
stats.append(f"range={df['timestamp'].min()} -> {df['timestamp'].max()}")
stats.append(f"duplicate_timestamps={int(df['timestamp'].duplicated().sum())}")
miss = df.isnull().sum()
miss = miss[miss > 0]
stats.append("missing=" + (miss.to_string() if len(miss) else "none"))
stats.append(f"aqi_skew={df['aqi'].skew()}")
stats.append(f"aqi_kurtosis={df['aqi'].kurtosis()}")
stats.append(f"aqi_mean={df['aqi'].mean()}")
stats.append(f"aqi_median={df['aqi'].median()}")
stats.append(f"aqi_std={df['aqi'].std()}")
iqr = df["aqi"].quantile(0.75) - df["aqi"].quantile(0.25)
outlier_hi = df["aqi"].quantile(0.75) + 1.5 * iqr
stats.append(f"aqi_iqr_outlier_hi={outlier_hi}")
stats.append(f"aqi_iqr_outlier_count={int((df['aqi'] > outlier_hi).sum())}")
stats.append("pollutant_skew=\n" + df[pollutants].skew().sort_values(ascending=False).to_string())
stats.append("corr_aqi=\n" + numeric_df.corr()["aqi"].sort_values(ascending=False).to_string())
adf = adfuller(df["aqi"].dropna())
stats.append(f"adf_stat={adf[0]}")
stats.append(f"adf_p={adf[1]}")
stats.append("yearly_mean=\n" + df.set_index("timestamp")["aqi"].resample("YE").mean().to_string())
stats.append("smog_describe=\n" + df.groupby("is_smog_season")["aqi"].describe().to_string())
stats.append("smog_mean=\n" + df.groupby("is_smog_season")["aqi"].mean().to_string())
stats.append("dtypes=\n" + df.dtypes.to_string())
(OUT / "eda_stats.txt").write_text("\n".join(stats), encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
sns.histplot(df["aqi"], kde=True, ax=axes[0], color="#1B365D")
axes[0].set_title("AQI Distribution")
sns.boxplot(x=df["aqi"], ax=axes[1], color="#1B365D")
axes[1].set_title("AQI Boxplot (outlier check)")
plt.tight_layout()
fig.savefig(OUT / "eda_aqi_dist.png", dpi=160, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
for ax, col in zip(axes.flat, pollutants):
    sns.histplot(df[col], kde=True, ax=ax, color="#2C5F8A")
    ax.set_title(f"{col} (skew={df[col].skew():.2f})")
for ax in axes.flat[len(pollutants) :]:
    ax.set_visible(False)
plt.tight_layout()
fig.savefig(OUT / "eda_pollutant_hist.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(11, 3.2))
plt.plot(df["timestamp"], df["aqi"], linewidth=0.4, color="#1B365D")
plt.title("AQI Over Time — Lahore (raw)")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig(OUT / "eda_aqi_time.png", dpi=160, bbox_inches="tight")
plt.close()

weather_cols = [c for c in ["temperature", "humidity", "wind_speed", "pressure"] if c in df.columns]
fig, axes = plt.subplots(1, len(weather_cols), figsize=(12, 3.2))
for ax, col in zip(axes, weather_cols):
    sns.scatterplot(x=df[col], y=df["aqi"], alpha=0.12, ax=ax, s=6, color="#1B365D")
    ax.set_title(f"AQI vs {col} (r={df[col].corr(df['aqi']):.2f})")
plt.tight_layout()
fig.savefig(OUT / "eda_aqi_weather.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap — Raw Columns")
plt.tight_layout()
plt.savefig(OUT / "eda_corr.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
plot_acf(df["aqi"].dropna(), lags=168, ax=axes[0])
plot_pacf(df["aqi"].dropna(), lags=72, ax=axes[1], method="ywm")
axes[0].set_title("ACF (lags up to 168h)")
axes[1].set_title("PACF (lags up to 72h)")
plt.tight_layout()
fig.savefig(OUT / "eda_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.close()

ts = df.set_index("timestamp")["aqi"].asfreq("h").interpolate()
decomposition = seasonal_decompose(ts, model="additive", period=24)
fig = decomposition.plot()
fig.set_size_inches(10.5, 7)
plt.tight_layout()
fig.savefig(OUT / "eda_decompose.png", dpi=140, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 4))
sns.boxplot(
    x=df["is_smog_season"].map({0: "Normal", 1: "Smog (Oct-Jan)"}),
    y=df["aqi"],
)
plt.title("AQI: Smog Season vs. Normal Season")
plt.xlabel("")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig(OUT / "eda_smog.png", dpi=160, bbox_inches="tight")
plt.close()

# AT8 comparison charts from exact notebook stdout
rows = [
    ("24h", "Persistence Baseline", 66.730405, 36.111431, 0.745204),
    ("24h", "XGBoost", 74.364967, 50.019281, 0.683567),
    ("24h", "LightGBM", 76.456676, 52.289149, 0.665515),
    ("24h", "Ensemble_top3_tabular", 77.085201, 53.331851, 0.659993),
    ("24h", "RandomForest", 84.914023, 59.164036, 0.587424),
    ("24h", "Ridge", 87.584091, 63.751444, 0.561069),
    ("24h", "GRU", 219.023846, 146.531186, -1.740176),
    ("24h", "LSTM", 232.032646, 156.863391, -2.075345),
    ("48h", "Persistence Baseline", 84.796609, 45.775352, 0.588437),
    ("48h", "XGBoost", 97.213262, 69.199877, 0.459083),
    ("48h", "Ensemble_top3_tabular", 98.112376, 70.283343, 0.449031),
    ("48h", "LightGBM", 101.157242, 72.808352, 0.414302),
    ("48h", "Ridge", 101.382767, 73.917026, 0.411688),
    ("48h", "RandomForest", 118.050540, 86.744712, 0.202344),
    ("48h", "LSTM", 234.142256, 152.548810, -2.132563),
    ("48h", "GRU", 246.401500, 157.766790, -2.469181),
    ("72h", "Persistence Baseline", 92.456124, 51.518059, 0.510480),
    ("72h", "Ridge", 104.353359, 77.654450, 0.376391),
    ("72h", "Ensemble_top3_tabular", 107.619632, 79.190920, 0.336742),
    ("72h", "XGBoost", 111.990073, 82.949948, 0.281779),
    ("72h", "LightGBM", 112.211034, 81.688058, 0.278942),
    ("72h", "RandomForest", 116.121902, 85.821522, 0.227804),
    ("72h", "LSTM", 235.968967, 160.365461, -2.182948),
    ("72h", "GRU", 239.031295, 159.196560, -2.266098),
]
# omit Prophet from bars (scale ~2100 would flatten others); table keeps Prophet
cmp = pd.DataFrame(rows, columns=["horizon", "model", "RMSE", "MAE", "R2"])
for h in ("24h", "48h", "72h"):
    sub = cmp[cmp["horizon"] == h].sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = ["#7A1F1F" if m == "Persistence Baseline" else "#1B365D" for m in sub["model"]]
    ax.barh(sub["model"], sub["RMSE"], color=colors)
    ax.set_xlabel("RMSE (absolute AQI)")
    ax.set_title(f"Test RMSE by model — {h} horizon (notebooks/02_training.ipynb)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUT / f"at8_rmse_{h}.png", dpi=160, bbox_inches="tight")
    plt.close()

print("figures written to", OUT)
