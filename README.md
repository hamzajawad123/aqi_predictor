# Pearls AQI Predictor

> 3-day-ahead Air Quality Index forecast for Lahore, built as a Feature / Training / Inference (FTI) pipeline for the 10Pearls Data Science Internship.

## Table of Contents

- [Introduction](#introduction)
- [Live app](#live-app)
- [Features](#features)
- [Technologies](#technologies)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Step-by-step setup](#step-by-step-setup)
- [Environment configuration](#environment-configuration)
- [Run the dashboard](#run-the-dashboard)
- [Run the API](#run-the-api)
- [Run the complete local stack](#run-the-complete-local-stack)
- [Docker](#docker)
  - [Option A — Docker Compose (recommended)](#option-a--docker-compose-recommended)
  - [Option B — Build and run the images yourself](#option-b--build-and-run-the-images-yourself)
- [Data, training, and notebooks](#data-training-and-notebooks)
- [GitHub Actions](#github-actions)
- [API endpoints](#api-endpoints)
- [Project structure](#project-structure)
- [Folder and file guide](#folder-and-file-guide)
- [Common commands](#common-commands)
- [Tests](#tests)
- [How to stop](#how-to-stop)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Useful links](#useful-links)
- [Contributing](#contributing)
- [License](#license)

---



## Introduction

**Pearls AQI Predictor** forecasts Lahore’s Air Quality Index at **24, 48, and 72 hours**.

Pollution concentrations come from the [OpenWeather Air Pollution API](https://openweathermap.org/api). Weather comes from Open-Meteo (no API key). Engineered features and trained models are stored in [Hopsworks](https://www.hopsworks.ai/). The dashboard is Streamlit (`app/Home.py`). FastAPI (`api/main.py`) is an optional HTTP wrapper around the same serving code.

The prediction target is a **continuous US EPA AQI from PM2.5** (see `src/utils/aqi_calculation.py`). OpenWeather’s native 1–5 `main.aqi` field is stored as `openweather_aqi_category` and is not the training target.

Default location in `src/config.py` / `.env.example`: **Lahore**, latitude `31.5497`, longitude `74.3436`.

---



## Live app

You can see the dashboard from here:

```markdown
[Open dashboard](https://YOUR-APP.streamlit.app)
```

---



## Features

- Hourly pollution + weather ingest, validation, and feature engineering
- Hopsworks feature group `aqi_features` (default version **4**) and model registry names `aqi_forecaster_{24,48,72}h`
- Models registered only when they beat a persistence baseline on **RMSE, MAE, and R²** (`src/training_pipeline.py`)
- Streamlit dashboard that reads Hopsworks through `src/utils/serving.py`
- Optional FastAPI routes: `GET /health`, `GET /predict`, `GET /model-metrics`
- GitHub Actions: hourly feature pipeline, daily training pipeline
- Docker images for the API and dashboard (`api/Dockerfile`, `app/Dockerfile`, `docker compose up --build`)

---



## Technologies


| Area                                          | What this repo uses                                                                 |
| --------------------------------------------- | ----------------------------------------------------------------------------------- |
| Language                                      | Python **3.11** (GitHub Actions and Dockerfiles)                                    |
| Dashboard                                     | Streamlit (`app/Home.py`, `app/requirements.txt`)                                   |
| Optional API                                  | FastAPI + Uvicorn (`api/main.py`, `api/requirements.txt`)                           |
| Feature store / registry                      | Hopsworks                                                                           |
| Pollution data                                | OpenWeather Air Pollution API                                                       |
| Weather data                                  | Open-Meteo                                                                          |
| Training stack (repo root `requirements.txt`) | scikit-learn, XGBoost, LightGBM, TensorFlow, Prophet, Optuna, SHAP                  |
| Tests                                         | pytest (`tests/`)                                                                   |
| Containers                                    | `docker-compose.yml`, `api/Dockerfile`, `app/Dockerfile`                            |
| CI                                            | `.github/workflows/feature_pipeline.yml`, `.github/workflows/training_pipeline.yml` |


There is **no local SQL database** in this repository. Features and models live in Hopsworks.

---



## Architecture

```
OpenWeather (pollution) ─┐
                          ├─▶ Feature Pipeline ──▶ Hopsworks Feature Store
Open-Meteo (weather)    ─┘   (GitHub Actions, hourly)        │
                                                             ▼
                                                      Training Pipeline ──▶ Hopsworks Model Registry
                                                      (GitHub Actions, daily)         │
                                                             ┌─────────────────────────┘
                                                             ▼
                                               src/utils/serving.py
                                                             │
                                    ┌────────────────────────┼────────────────────────┐
                                    ▼                                                 ▼
                          Streamlit (app/Home.py)                          FastAPI (api/main.py)
                          talks to Hopsworks directly                    optional GET /predict wrapper
```

---



## Prerequisites

Install these before you clone.


| Tool                                     | Required?                            | Version in this repo                           | Check                                                              |
| ---------------------------------------- | ------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------ |
| Git                                      | Yes                                  | Not pinned                                     | `git --version`                                                    |
| Python                                   | Yes                                  | **3.11** (Actions + Docker `python:3.11-slim`) | `python --version`                                                 |
| pip                                      | Yes (comes with Python)              | Not pinned                                     | `python -m pip --version`                                          |
| OpenWeather API key                      | Yes for fetch / hourly / backfill    | —                                              | Create at [openweathermap.org/api](https://openweathermap.org/api) |
| Hopsworks project + API key              | Yes for store, train, dashboard, API | Host default `eu-west.cloud.hopsworks.ai`      | [hopsworks.ai](https://www.hopsworks.ai/)                          |
| Docker Desktop / Docker Engine + Compose | Only if you use Compose              | Compose file `version: "3.9"`                  | `docker --version` then `docker compose version`                   |
| VS Code                                  | Optional                             | —                                              | —                                                                  |


Open-Meteo does not require a key.

On Windows, use **Command Prompt**, **PowerShell**, or the **VS Code terminal**. Run clone/install/run commands from the folder where you want the project (or from the repo root after clone).

---



## Quick Start

Shortest path: clone, Python 3.11 venv, dashboard dependencies, `.env`, Streamlit.

```bash
git clone https://github.com/hamzajawad123/aqi_predictor.git
cd aqi_predictor
python -m venv .venv
```

**Windows (Command Prompt / PowerShell / VS Code terminal):**

```bat
.venv\Scripts\activate
python -m pip install -r app\requirements.txt
copy .env.example .env
```

**macOS / Linux:**

```bash
source .venv/bin/activate
python -m pip install -r app/requirements.txt
cp .env.example .env
```

Edit `.env` in the **repository root** (see [Environment configuration](#environment-configuration)). Then:

```bash
streamlit run app/Home.py
```

Streamlit’s default URL (not overridden in `.streamlit/config.toml`):

```text
http://localhost:8501
```

You need a Hopsworks project that already has feature group **v4** and registered models. If the store is empty, follow [Data, training, and notebooks](#data-training-and-notebooks) first.

To run from Docker images instead of a local Python venv, see [Docker](#docker).

---



## Step-by-step setup



### Step 1 — Install required software

1. Install Git.
2. Install **Python 3.11**. Confirm with `python --version`.
3. Create an OpenWeather API key (air pollution).
4. Create a Hopsworks project and API key. Confirm the host in your Hopsworks URL (this repo defaults to `eu-west.cloud.hopsworks.ai` in `src/config.py`).
5. Optional: install Docker Desktop if you will run `docker compose`.
6. Optional: install [VS Code](https://code.visualstudio.com/).



### Step 2 — Clone the repository

In Command Prompt, PowerShell, or a VS Code terminal, `cd` to the parent folder where you want the project, then:

```bash
git clone https://github.com/hamzajawad123/aqi_predictor.git
cd aqi_predictor
```

GitHub creates a folder named `aqi_predictor` (underscore). The clone URL is `https://github.com/hamzajawad123/aqi_predictor`.

### Step 3 — Open the project in VS Code (optional)

1. Start VS Code.
2. **File → Open Folder…** and select the cloned `aqi_predictor` folder.
3. Confirm the explorer root contains `README.md`, `app/`, `src/`, and `.env.example`.
4. Open a terminal: **Terminal → New Terminal** (or `Ctrl+``).
5. The terminal working directory should be the repo root.

If the VS Code `code` CLI is installed:

```bash
cd aqi_predictor
code .
```



### Step 4 — Create a virtual environment and install dependencies

From the **repository root**, with Python 3.11:

```bash
python -m venv .venv
```

Activate:

```bat
.venv\Scripts\activate
```

```bash
source .venv/bin/activate
```

**Dashboard only** (Streamlit Cloud uses this file):

```bash
python -m pip install -r app/requirements.txt
```

**Optional API** (same packages as `api/Dockerfile`):

```bash
python -m pip install -r api/requirements.txt
```

**Full project** (feature pipeline, training, notebooks, pytest — includes TensorFlow and Prophet):

```bash
python -m pip install -r requirements.txt
```

Do **not** use the repo-root `requirements.txt` as the Streamlit Community Cloud requirements file. That file includes TensorFlow/Prophet for training. Cloud deploy must use `[app/requirements.txt](app/requirements.txt)`.

### Step 5 — Configure environment variables

See the next section. Create `.env` in the **repository root**, not inside `app/` or `api/`.

### Step 6 — Start the dashboard

See [Run the dashboard](#run-the-dashboard). FastAPI is optional for `app/Home.py`.

---



## Environment configuration

1. Copy the example file in the **repository root**:

**Windows:**

```bat
copy .env.example .env
```

**macOS / Linux:**

```bash
cp .env.example .env
```

1. Open `.env` and put **your** keys. Never commit `.env` (it is listed in `.gitignore`).

Variables from `[.env.example](.env.example)` and defaults from `[src/config.py](src/config.py)`:


| Variable                 | Required?                                                           | Role                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `OPENWEATHER_API_KEY`    | Required for fetch / hourly / backfill (`config.validate_config()`) | OpenWeather                                                                                                                        |
| `HOPSWORKS_API_KEY`      | Required for Hopsworks                                              | Feature store and registry                                                                                                         |
| `HOPSWORKS_PROJECT_NAME` | Required for Hopsworks                                              | Feature store and registry                                                                                                         |
| `HOPSWORKS_HOST`         | Optional                                                            | Not in `.env.example`. `src/config.py` default: `eu-west.cloud.hopsworks.ai`                                                       |
| `CITY_NAME`              | Optional                                                            | Default `Lahore`                                                                                                                   |
| `LATITUDE`               | Optional                                                            | Default `31.5497`                                                                                                                  |
| `LONGITUDE`              | Optional                                                            | Default `74.3436`                                                                                                                  |
| `DATA_START_DATE`        | Optional                                                            | Default `2020-11-27`                                                                                                               |
| `RAW_DATA_PATH`          | Optional                                                            | Default `data/raw/aqi_raw_merged.parquet`                                                                                          |
| `FEATURE_GROUP_NAME`     | Optional                                                            | Default `aqi_features`                                                                                                             |
| `FEATURE_GROUP_VERSION`  | Optional                                                            | Default `4`                                                                                                                        |
| `FEATURE_VIEW_NAME`      | Optional                                                            | Default `aqi_feature_view`                                                                                                         |
| `MODEL_NAME`             | Optional                                                            | Default `aqi_forecaster`                                                                                                           |
| `TRAIN_START_DATE`       | Optional                                                            | Default `2025-04-04`. Empty string trains on all history                                                                           |
| `USE_DELTA_SHRINKAGE`    | Optional                                                            | Default `true`                                                                                                                     |
| `API_BASE_URL`           | Compose / legacy                                                    | `.env.example` value `http://api:8000` (Docker service name). `app/Home.py` **does not read this**; it uses `src/utils/serving.py` |


Example shape (use your own secrets):

```env
OPENWEATHER_API_KEY=your_openweather_api_key_here
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT_NAME=your_hopsworks_project_name_here
FEATURE_GROUP_VERSION=4
```

`src/config.py` `validate_config()` fails if `OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`, or `HOPSWORKS_PROJECT_NAME` is missing.

### Streamlit Cloud secrets

Deploy at [share.streamlit.io](https://share.streamlit.io):

- Main file: `app/Home.py`
- Requirements: `app/requirements.txt`
- Python: 3.11

Secrets (same names as `.env`; do not paste real keys into git):

```toml
HOPSWORKS_API_KEY = "your_hopsworks_api_key_here"
HOPSWORKS_PROJECT_NAME = "your_hopsworks_project_name_here"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
FEATURE_GROUP_VERSION = "4"
CITY_NAME = "Lahore"
```

Locally, `app/bootstrap.py` can also load `.streamlit/secrets.toml` if that file exists. Prefer `.env` for local CLI runs (`load_dotenv()` in `src/config.py`).

---



## Run the dashboard

**Where:** repository root, venv activated, `.env` filled, `app/requirements.txt` (or full `requirements.txt`) installed.

**Terminal 1 — Streamlit**

```bash
streamlit run app/Home.py
```

**Browser**

```text
http://localhost:8501
```

Docker sets `--server.port=8501` and `--server.address=0.0.0.0` in `app/Dockerfile`. A local `streamlit run` uses Streamlit’s default port **8501** (`.streamlit/config.toml` does not set a port).

`app/client.py` calls `src.utils.serving.dashboard_state()` (Hopsworks). You do **not** need Uvicorn running for this page.

Leave this terminal open while you use the app.

---



## Run the API

Optional. Same serving logic as the dashboard (`api/main.py` wraps `src.utils.serving`).

**Where:** repository root, venv activated, `api/requirements.txt` or full `requirements.txt` installed, Hopsworks variables set.

**Terminal 2 — FastAPI** (separate from Streamlit if both run)

```bash
uvicorn api.main:app --reload
```

Uvicorn’s default bind is port **8000**. `api/Dockerfile` uses `--host 0.0.0.0 --port 8000`.


| Check         | URL                                   |
| ------------- | ------------------------------------- |
| Health        | `http://localhost:8000/health`        |
| Forecast      | `http://localhost:8000/predict`       |
| Model metrics | `http://localhost:8000/model-metrics` |


This project does not set `docs_url=None` on `FastAPI()`, so FastAPI’s default OpenAPI UI is available at `http://localhost:8000/docs` while Uvicorn is running.

---



## Run the complete local stack

Two processes if you want both UI and HTTP API. The dashboard does not call the API.

### Terminal 1 — API (optional)

```bash
uvicorn api.main:app --reload
```

```text
http://localhost:8000/health
```



### Terminal 2 — Dashboard

```bash
streamlit run app/Home.py
```

```text
http://localhost:8501
```

Hopsworks must already contain features and registered models for forecasts to appear.

---



## Docker

You can run this project from **Docker images** instead of installing Python packages on your machine. The images are defined in [`api/Dockerfile`](api/Dockerfile) (FastAPI, port 8000) and [`app/Dockerfile`](app/Dockerfile) (Streamlit, port 8501). Both start from `python:3.11-slim`.

A published Docker Hub image name is **not recorded** in this repository, so there is no `docker pull user/name` URL to copy yet. The procedure below **builds the images from this repo**, then runs them. You still need Docker Desktop (Windows/macOS) or Docker Engine + Compose, a clone of the repo, and a root `.env` (Hopsworks keys at minimum for the dashboard).

Confirm Docker:

```bash
docker --version
docker compose version
```

Clone (same as [Step 2](#step-2--clone-the-repository)), `cd aqi_predictor`, and copy `.env.example` to `.env`.

### Option A — Docker Compose (recommended)

From the **repository root**, with `.env` present (`docker-compose.yml` uses `env_file: .env`):

```bash
docker compose up --build
```

This builds both images and starts both containers. The first run downloads `python:3.11-slim` if it is not already on your machine.

| Service | Container name | Host port | Image command |
| ------- | -------------- | --------- | ------------- |
| `api`   | `aqi_api`      | **8000**  | `uvicorn api.main:app --host 0.0.0.0 --port 8000` |
| `app`   | `aqi_app`      | **8501**  | `streamlit run app/Home.py --server.port=8501 --server.address=0.0.0.0` |

Compose sets `API_BASE_URL=http://api:8000` on the `app` service. `app/Home.py` currently does not use that variable.

Open:

```text
Dashboard:  http://localhost:8501
API health: http://localhost:8000/health
```

Healthchecks inside the images:

- API: `http://localhost:8000/health`
- App: `http://localhost:8501/_stcore/health`

Run in the background:

```bash
docker compose up --build -d
```

Stop and remove the containers:

```bash
docker compose down
```

### Option B — Build and run the images yourself

Use this if you want the Docker images without Compose, or you only want the dashboard image.

From the **repository root** (build context is `.` because both Dockerfiles `COPY src/` from the repo root):

**1. Build**

```bash
docker build -f api/Dockerfile -t aqi-api:local .
docker build -f app/Dockerfile -t aqi-app:local .
```

**2. Confirm the images exist**

```bash
docker images aqi-api:local
docker images aqi-app:local
```

**3. Run the dashboard image**

```bash
docker run --rm --name aqi_app --env-file .env -p 8501:8501 aqi-app:local
```

```text
http://localhost:8501
```

**4. Run the API image** (optional, separate terminal)

```bash
docker run --rm --name aqi_api --env-file .env -p 8000:8000 aqi-api:local
```

```text
http://localhost:8000/health
```

**5. Stop**

In the terminal running the container: **Ctrl+C**.

Or from another terminal:

```bash
docker stop aqi_app
docker stop aqi_api
```

`--env-file .env` passes the same variables Compose uses. `--rm` deletes the container when it stops; the **images** stay until you run `docker rmi aqi-app:local aqi-api:local`.

---



## Data, training, and notebooks

Run these from the **repository root** with the **full** `[requirements.txt](requirements.txt)` and a complete `.env`.

### Fetch raw history (no Hopsworks write)

```bash
python -m src.feature_pipeline raw-snapshot
```

Optional start date: `python -m src.feature_pipeline raw-snapshot YYYY-MM-DD`.

Writes `data/raw/aqi_raw_merged.parquet` (path overridable with `RAW_DATA_PATH`).

### Explore

Open `[notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)` after the parquet exists. See `[notebooks/README.md](notebooks/README.md)`.

```bash
jupyter notebook
```

(`jupyter` is listed in the root `requirements.txt`.)

### Push features to Hopsworks

```bash
python -m src.feature_pipeline push-features
python -m src.utils.hopsworks_utils
```

`push-features` builds the engineered table from the local parquet into `aqi_features` v4 (or `FEATURE_GROUP_VERSION`). The second command creates the feature view used by training.

### Hourly-style update (lookback window)

```bash
python -m src.feature_pipeline
```



### Historical backfill (fetch + engineer + insert)

```bash
python -m src.feature_pipeline backfill
```

Optional: `python -m src.feature_pipeline backfill YYYY-MM-DD`.

### Train

```bash
python -m src.training_pipeline
```

Trains 24 / 48 / 72 hour models, scores against persistence, and registers `aqi_forecaster_{24,48,72}h` only when a model beats persistence on RMSE, MAE, and R².

GPU training can also be run from `[notebooks/02_training.ipynb](notebooks/02_training.ipynb)`. Training artefacts are described in `[reports/README.md](reports/README.md)`.

---



## GitHub Actions

Add **repository secrets**: `OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`. Workflows also read `FEATURE_VIEW_NAME` from secrets if you set it.


| Workflow                                                                             | Trigger                                        | Command                           |
| ------------------------------------------------------------------------------------ | ---------------------------------------------- | --------------------------------- |
| `[.github/workflows/feature_pipeline.yml](.github/workflows/feature_pipeline.yml)`   | Hourly (`0 * * * *`) and **workflow_dispatch** | `python -m src.feature_pipeline`  |
| `[.github/workflows/training_pipeline.yml](.github/workflows/training_pipeline.yml)` | Daily 02:00 UTC and **workflow_dispatch**      | `python -m src.training_pipeline` |


Both pin `FEATURE_GROUP_VERSION` to **4**. Python **3.11**. Jobs install `requirements.txt`.

---



## API endpoints

Defined in `[api/main.py](api/main.py)`. No request body. No auth in this file.


| Method | Path             | Purpose                                                                                                              |
| ------ | ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| GET    | `/health`        | Returns `{"status": "ok"}`                                                                                           |
| GET    | `/predict`       | Forecast payload (`ForecastResponse`: city, per-horizon forecasts, `forecast_72h`, `hazardous_alert`, `current_aqi`) |
| GET    | `/model-metrics` | Metrics (or error) per loaded registry model                                                                         |


Hazardous flag threshold in `src/utils/serving.py`: `HAZARDOUS_THRESHOLD = 151`.

---



## Project structure

```text
aqi_predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml
│   └── training_pipeline.yml
├── .streamlit/
│   └── config.toml
├── api/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── app/
│   ├── Dockerfile
│   ├── Home.py
│   ├── bootstrap.py
│   ├── charts.py
│   ├── client.py
│   ├── theme.py
│   └── requirements.txt
├── data/                      # gitignored parquet; created by raw-snapshot
├── notebooks/
│   ├── README.md
│   ├── 01_eda.ipynb
│   └── 02_training.ipynb
├── reports/
│   └── README.md
├── src/
│   ├── config.py
│   ├── feature_pipeline.py
│   ├── training_pipeline.py
│   ├── train_prophet.py
│   ├── train_lstm.py
│   ├── train_gru.py
│   └── utils/
│       ├── aqi_calculation.py
│       ├── data_fetch.py
│       ├── data_validation.py
│       ├── evaluation.py
│       ├── feature_engineering.py
│       ├── hopsworks_utils.py
│       ├── serving.py
│       ├── optuna_tuning.py
│       ├── raw_io.py
│       └── sequences.py
├── tests/
│   ├── test_aqi_calculation.py
│   ├── test_data_validation.py
│   ├── test_evaluation.py
│   ├── test_feature_engineering.py
│   └── test_feature_pipeline.py
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── README.md
└── requirements.txt
```

---



## Folder and file guide


| Path                                                     | Purpose                                             |
| -------------------------------------------------------- | --------------------------------------------------- |
| `[app/Home.py](app/Home.py)`                             | Streamlit dashboard entry                           |
| `[app/client.py](app/client.py)`                         | Cached `dashboard_state()` from Hopsworks           |
| `[app/bootstrap.py](app/bootstrap.py)`                   | Adds repo root to `sys.path`; optional secrets.toml |
| `[api/main.py](api/main.py)`                             | FastAPI app                                         |
| `[src/config.py](src/config.py)`                         | Env loading and defaults                            |
| `[src/feature_pipeline.py](src/feature_pipeline.py)`     | `raw-snapshot`, `backfill`, `push-features`, hourly |
| `[src/training_pipeline.py](src/training_pipeline.py)`   | Train + persistence gate + registry                 |
| `[src/utils/serving.py](src/utils/serving.py)`           | Model load, `/predict` body, dashboard payload      |
| `[src/utils/data_fetch.py](src/utils/data_fetch.py)`     | OpenWeather + Open-Meteo                            |
| `[.env.example](.env.example)`                           | Env template                                        |
| `[docker-compose.yml](docker-compose.yml)`               | API :8000 + app :8501                               |
| `[reports/final_report.docx](reports/final_report.docx)` | Project write-up (if present in your clone)         |
| `[tests/](tests/)`                                       | pytest                                              |


---



## Common commands


| Command                                         | What it does                       |
| ----------------------------------------------- | ---------------------------------- |
| `python -m pip install -r app/requirements.txt` | Dashboard dependencies             |
| `python -m pip install -r api/requirements.txt` | API dependencies                   |
| `python -m pip install -r requirements.txt`     | Full training + notebooks + pytest |
| `streamlit run app/Home.py`                     | Start dashboard                    |
| `uvicorn api.main:app --reload`                 | Start API on port 8000             |
| `python -m src.feature_pipeline raw-snapshot`   | Local raw parquet                  |
| `python -m src.feature_pipeline push-features`  | Features → Hopsworks               |
| `python -m src.utils.hopsworks_utils`           | Create feature view                |
| `python -m src.feature_pipeline`                | Hourly feature job                 |
| `python -m src.feature_pipeline backfill`       | Historical Hopsworks insert        |
| `python -m src.training_pipeline`               | Train and maybe register           |
| `docker compose up --build`                     | Build images from this repo and start API + dashboard |
| `docker compose up --build -d`                  | Same, detached                     |
| `docker compose down`                           | Stop Compose stack                 |
| `docker build -f api/Dockerfile -t aqi-api:local .` | Build API image                |
| `docker build -f app/Dockerfile -t aqi-app:local .` | Build dashboard image          |
| `docker run --rm --env-file .env -p 8501:8501 aqi-app:local` | Run dashboard image |
| `docker run --rm --env-file .env -p 8000:8000 aqi-api:local` | Run API image |
| `pytest tests/`                                 | Run tests                          |


---



## Tests

From the repository root, after installing root `requirements.txt` (includes `pytest==8.3.3`):

```bash
pytest tests/
```

---



## How to stop

- Streamlit or Uvicorn in a terminal: **Ctrl+C**
- Docker Compose: `docker compose down` from the repo root
- Docker images started with `docker run --name aqi_app` / `aqi_api`: `docker stop aqi_app` and `docker stop aqi_api`

---



## Troubleshooting

**Missing** `.env` **/** `EnvironmentError: Missing required environment variables`  
`src/config.py` `validate_config()` requires `OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`, and `HOPSWORKS_PROJECT_NAME`. Copy `.env.example` to `.env` in the repo root.

**Dashboard or API cannot reach Hopsworks**  
Check host, project name, and API key. If login hits the wrong cluster, set `HOPSWORKS_HOST` to the host in your Hopsworks URL (`src/config.py` default is `eu-west.cloud.hopsworks.ai`).

**Empty or failed forecasts**  
The serving layer reads the feature group and registry. Run feature ingest and training (or confirm v4 + `aqi_forecaster_{24,48,72}h` exist) before expecting dashboard numbers.

**Streamlit Cloud install is huge or fails**  
Set the requirements file to `app/requirements.txt`, not the repo-root `requirements.txt`.

**Wrong feature group version**  
Default and GitHub Actions pin **4**. An old `FEATURE_GROUP_VERSION` secret can write to a pre-delta group; `hopsworks_utils.py` explains that failure.

**Port 8501 or 8000 already in use** (general)  
Stop the other process or change the port in the run command. This repo’s Dockerfiles use 8501 and 8000.

`docker compose` **cannot start**  
Compose requires a root `.env` (`env_file: .env`). Docker Desktop (or Engine + Compose plugin) must be running.

**Python version**  
Use 3.11 to match Actions and `python:3.11-slim`.

**venv not active**  
Windows: `.venv\Scripts\activate`. If `python` is not found, use `py -3.11`.

---



## Security

Never commit API keys, tokens, or `.env` to GitHub. `.gitignore` already ignores `.env` and `.env.*` except `.env.example`.

There is no `SECURITY.md` in this repository.

---



## Useful links


|                   |                                                                                                                |
| ----------------- | -------------------------------------------------------------------------------------------------------------- |
| Source repository | [https://github.com/hamzajawad123/aqi_predictor](https://github.com/hamzajawad123/aqi_predictor)               |
| Issue tracker     | [https://github.com/hamzajawad123/aqi_predictor/issues](https://github.com/hamzajawad123/aqi_predictor/issues) |
| Live app          | Same as [Live app](#live-app) (Streamlit Community Cloud URL after deploy)                                     |
| OpenWeather API   | [https://openweathermap.org/api](https://openweathermap.org/api)                                               |
| Hopsworks         | [https://www.hopsworks.ai/](https://www.hopsworks.ai/)                                                         |
| Project report    | `[reports/final_report.docx](reports/final_report.docx)`                                                       |
| Notebook notes    | `[notebooks/README.md](notebooks/README.md)`                                                                   |


---



## Contributing

This repository does not include a `CONTRIBUTING.md`. Suggested workflow (not project-specific policy):

1. Create a branch.
2. Make changes.
3. Run `pytest tests/` when your change touches tested code.
4. Commit (do not commit `.env` or keys).
5. Push and open a pull request.

---



## License

No `LICENSE` file is present in this repository. Do not assume redistribution terms until the maintainers add one.